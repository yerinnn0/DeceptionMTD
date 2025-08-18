

from .IRL import IRL
from itertools import product


import numpy as np
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import torch
import torch.nn as nn


class DeepMaxEntIRL(IRL):
    
    def __init__(self, mmdp, feature_map = "identity", epochs=100, learning_rate=0.0001, layers = (128,64)):
        super(DeepMaxEntIRL, self).__init__(mmdp, feature_map)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        def sparse_scipy_to_torch(coo):
            values = torch.tensor(coo.data, dtype=torch.float32)
            indices = torch.vstack((torch.tensor(coo.row), torch.tensor(coo.col))).long()
            shape = coo.shape
            return torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float32)
        
        self.epochs = epochs
        self.learning_rate = learning_rate
        
        self.feature_matrix = sparse_scipy_to_torch(self.feature_matrix).to(self.device)
        self.discount = mmdp.gamma

        self.rews = []
        self.net = []
        last = self.feature_matrix.shape[1]  # feature_dim
        for l in layers:
            self.net.append(nn.Linear(last, l))
            self.net.append(nn.ReLU())
            last = l
        self.net.append(nn.Linear(last, 1))
        self.net = nn.Sequential(*self.net)
        self.net.to(self.device)
        self.optim = torch.optim.Adam(self.net.parameters(), self.learning_rate)
        
        
    def forward(self, features):
        x = self.net(features)  # input_shape: (n_states_actions, feature_dim)
        return (x - x.mean()) / x.std()
        # return x  # output_shape: (n_states_actions, )

    def irl(self, trajectories):
        """
        Find the reward function for the given trajectories.

        feature_matrix: Matrix with the nth row representing the nth state. np.array with shape (n_states*n_action, 1)
        n_actions: Number of actions. int.
        discount: Discount factor of the MMDP. float.
        transition_probability: Transition probability matrix of MMDP. np.array with shape (n_joint_states, n_joint_actions, n_joint_states).
        trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
        epochs: Number of gradient descent steps. int.
        learning_rate: Gradient descent learning rate. float.
        return: Reward vector with shape (n_states * n_actions,).
        """
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        feature_matrix = self.feature_matrix.to(self.device)
        
        print("Running IRL with Deep MaxEnt")

        # Calculate the feature expectations \tilde{phi}.
        # feature_expectations = self.find_feature_expectations(trajectories) # (n_states * n_actions, )
        svf = self.find_svf(trajectories)
        svf = torch.tensor(svf, dtype=torch.float32, device=self.device)
        if not torch.is_tensor(trajectories):
            trajectories_torch = torch.tensor(trajectories, dtype=torch.float32, device=self.device)
        else:
            trajectories_torch = trajectories.to(self.device)
        # trajectories : (traj_number, steps, 2)

        # Gradient descent on alpha.
        for i in range(self.epochs):
            print("epoch:", i)
            # r = self.feature_matrix.dot(alpha)
            r = self.forward(feature_matrix).flatten()   # (n_states * n_actions, )
            scale = torch.sum(torch.abs(r))
            # r = r/ scale
            expected_svf = self.find_expected_svf(r, trajectories_torch)  # (n_states * n_actions, )
            if not torch.is_tensor(expected_svf):
                expected_svf = torch.from_numpy(expected_svf).float()
            expected_svf = expected_svf.to(self.device)
            
            grad = (svf - expected_svf) # (n_states*n_actions, )
            _, policy = self.value_iteration(r, 1e-5)
            policy = policy.reshape(1, -1).float().to(self.device)
            loss = -torch.matmul(torch.log(policy + 1e-6), svf)
            
            self.optim.zero_grad()
            r.backward(gradient =-grad)
            self.optim.step()
            print("loss:", loss, "grad:", torch.mean(grad), "grad_norm:", torch.norm(svf - expected_svf), scale)
            
        # return self.feature_matrix.dot(alpha).reshape((n_states * n_actions,))  # (n_states * n_actions, )
        return r.detach().cpu().numpy().reshape((n_states * n_actions,))

    def find_feature_expectations(self, trajectories: torch.Tensor) -> torch.Tensor:
        """
        Compute the feature expectations using dense PyTorch tensors.

        Args:
            trajectories: Tensor of shape (num_traj, len_traj, 2), where each element is (state, action)

        Returns:
            feature_expectations: Tensor of shape (feature_dim,)
        """
        n_actions = self.mmdp.n_joint_actions
        feature_dim = self.feature_matrix.shape[1]
        feature_expectations = torch.zeros(feature_dim, dtype=torch.float32, device = self.device)

        if len(trajectories.shape) == 4:
            trajectories = trajectories[0]  # shape: (num_traj, len_traj, 2)

        for traj in trajectories:  # shape: (len_traj, 2)
            for step in traj:
                state = step[0].item()
                action = step[1].item()
                row_index = int(state * n_actions + action)
                feature_expectations += self.feature_matrix[row_index]

        feature_expectations /= trajectories.shape[0]

        return feature_expectations   # (feature_dim, )


    def find_expected_svf_cpu(self, r, trajectories):
        """
        Find the expected state visitation frequencies using algorithm 1 from
        Ziebart et al. 2008.

        n_states: Number of states. int.
        r: Reward. NumPy array with shape (n_states*n_actions,).
        n_actions: Number of actions. int.
        discount: Discount factor of the MMDP. float.
        transition_probability: Transition probability matrix of MMDP. np.array with shape (n_joint_states, n_joint_actions, n_joint_states).
        trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
        
        return: Expected state visitation frequencies vector with shape # (n_states * n_action, ).
        """
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        discount = self.mmdp.gamma
        transition_probability = self.mmdp.transition_matrices
        
        if len(trajectories.shape) == 4:
            trajectories = trajectories[0,:,:,:]

        n_trajectories = trajectories.shape[0]
        trajectory_length = trajectories.shape[1]
        
        policy = self.value_iteration(r.reshape(n_states,-1), theta = 1e-5)[1]# (n_states,)
        
        start_state_action_count = torch.zeros(n_states * n_actions) # Initial Distribution
        for trajectory in trajectories: # (step, 2)
            state = int(trajectory[0, 0])
            action = int(trajectory[0, 1])
            start_state_action_count[state*n_actions + action] += 1
        p_start_state_action = start_state_action_count/n_trajectories

        expected_svf = torch.tile(p_start_state_action, (trajectory_length, 1)).T # (n_states * n_action, step, 1)
        
        for t in range(1, trajectory_length):
            expected_svf[:, t] = 0 # (n_states * n_action, 1)
            for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
                expected_svf[i*n_actions + j, t] += (torch.sum(expected_svf[k*n_actions:(k+1)*n_actions , t-1]) *
                                    policy[i, j] * transition_probability[i, j, k]) * discount

        return expected_svf.sum(axis=1) # (n_states * n_action, )



    # def find_expected_svf(self, r, trajectories):
    #     """
    #     Compute expected state-action visitation frequencies (SVF) on GPU.

    #     Args:
    #         r: reward vector, shape (n_states * n_actions,)
    #         trajectories: tensor or ndarray of shape (num_traj, traj_len, 2)
    #                     each element is [state, action]

    #     Returns:
    #         expected_svf: tensor of shape (n_states * n_actions,)
    #     """
    #     n_states = self.mmdp.n_joint_states
    #     n_actions = self.mmdp.n_joint_actions
    #     discount = self.mmdp.gamma
    #     T = self.mmdp.transition_matrices  # shape (s, a, s')

    #     # Convert to torch tensors on GPU if needed
    #     if isinstance(T, torch.Tensor):
    #         T = T.to(self.device)
    #     else:
    #         T = torch.tensor(T, dtype=torch.float32, device=self.device)

    #     if isinstance(r, torch.Tensor):
    #         r = r.to(self.device)
    #     else:
    #         r = torch.tensor(r, dtype=torch.float32, device=self.device)

    #     if not torch.is_tensor(trajectories):
    #         trajectories = torch.tensor(trajectories, dtype=torch.long, device=self.device)
    #     else:
    #         trajectories = trajectories.to(self.device)

    #     # Compute policy from reward via value iteration (policy shape: (n_states, n_actions))
    #     _, policy = self.value_iteration(r.view(n_states, n_actions))

    #     n_traj = trajectories.shape[0]
    #     traj_len = trajectories.shape[1]

    #     # Compute initial state-action distribution from trajectories
    #     start_sa = torch.zeros(n_states * n_actions, device=self.device)
    #     for traj in trajectories:
    #         s0 = traj[0, 0].item()
    #         a0 = traj[0, 1].item()
    #         start_sa[s0 * n_actions + a0] += 1
    #     start_sa /= n_traj

    #     # Initialize expected SVF tensor: shape (traj_len, n_states * n_actions)
    #     expected_svf = torch.zeros((traj_len, n_states * n_actions), device=self.device)
    #     expected_svf[0] = start_sa

    #     # Reshape policy for easy indexing: (n_states, n_actions)
    #     # T has shape (s, a, s') and policy shape (s, a)
    #     # We'll propagate expected_svf over time steps
    #     for t in range(1, traj_len):
    #         prev = expected_svf[t - 1].view(n_states, n_actions)  # (s, a)
    #         # For each next state s', sum over s,a:
    #         # expected_svf[t, s'*n_actions:(s'+1)*n_actions] = policy[s'] * sum_{s,a} prev[s,a] * policy[s,a] * T[s,a,s']
    #         expected_svf_t = torch.zeros(n_states, n_actions, device=self.device)

    #         # Compute transition probabilities weighted by previous visitation and policy
    #         # Shape explanation:
    #         # prev: (s,a), policy: (s,a), T: (s,a,s')
    #         # We want to compute contribution to s' from all (s,a)
    #         # Then multiply by policy at s'
    #         # We'll do this efficiently by batch matrix multiply

    #         # Step 1: compute weighted transition from previous expected SVF
    #         # prev * policy shape (s,a)
    #         weighted_prev = prev * policy  # (s,a)

    #         # Step 2: sum over (s,a) contributions to each s'
    #         # For s', compute: sum_{s,a} weighted_prev[s,a] * T[s,a,s']
    #         contrib_s_prime = torch.einsum('sa,sas->s', weighted_prev, T)  # (s')

    #         # Step 3: multiply by policy[s'] for each action
    #         expected_svf_t = policy * contrib_s_prime.unsqueeze(1)  # (s', a)

    #         # Apply discount factor gamma
    #         expected_svf[t] = (discount * expected_svf_t).view(-1)

    #     # Sum expected SVF over all timesteps
    #     return expected_svf.sum(dim=0)
