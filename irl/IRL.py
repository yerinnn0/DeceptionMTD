
import time
import numpy as np
import scipy.sparse as sp

from itertools import product

import torch

class IRL:
    
    def __init__(self, mmdp, feature_map = "identity"):
        self.mmdp = mmdp
        self.expected_rewards = None
        self.estimated_policy = None
        self.estimated_value_function = None
        self.self_estimated_value_function = None

        print("Setting feature matrix")
        if feature_map == "identity":
            # self.feature_matrix = np.eye(mmdp.n_joint_states*mmdp.n_joint_actions)
            self.feature_matrix = sp.identity(mmdp.n_joint_states*mmdp.n_joint_actions, format='coo')
        else:
            self.feature_matrix = sp.csr_matrix(mmdp.feature_matrix(feature_map = feature_map)) # Numpy array

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("Converting transition matrix")

        self.transition_matrix_torch = torch.tensor(mmdp.transition_matrices, dtype=torch.float32, device=self.device)  # (s', a, s)


    def generate_trajectories_cpu(self, policy, num_traj, len_traj, reset_s0 = True):
        """
        Generates trajectories based on given policy in MMDP.
        
        policy: Stochastic policy of MMDP. np.array(n_states, n_actions).
        num_traj: Number of trajectory to generate. int.
        len_traj: Length of trajectory to generate. int
        reset_s0: Whether to reset the initial state. bool.
        """

        trajectories = np.zeros((num_traj, len_traj, 2))

        for k in range(num_traj):
            # if k % 100 == 0:
            #     print(k)
            if reset_s0:
                self.mmdp.reset_initial_state()
            trajectories[k, :, 0], trajectories[k, :, 1], _ = self.mmdp.run_MApolicy_cpu(policy, len_traj)

        self.trajectories = trajectories

        return trajectories
    
    
    def generate_trajectories(self, policy, num_traj, len_traj, reset_s0 = False):
        """
        Generates trajectories based on given policy in MMDP.
        
        policy: Stochastic policy of MMDP. np.array(n_states, n_actions).
        num_traj: Number of trajectory to generate. int.
        len_traj: Length of trajectory to generate. int
        reset_s0: Whether to reset the initial state. bool.
        """
        if not isinstance(policy, torch.Tensor):
            policy_t = torch.as_tensor(policy, dtype=torch.float32, device=self.device)
        else:
            if policy.device != self.device or policy.dtype != torch.float32:
                policy_t = policy.to(device=self.device, dtype=torch.float32)
            else:
                policy_t = policy

        trajectories = torch.zeros((num_traj, len_traj, 2), dtype=torch.long, device=self.device)

        for k in range(num_traj):
            # if k % 100 == 0:
            print(k)
            if reset_s0:
                self.mmdp.reset_initial_state()
            # time0 = time.time()
            state_traj, action_traj, _ = self.mmdp.run_MApolicy(policy_t, len_traj)
            trajectories[k, :, 0] = state_traj
            trajectories[k, :, 1] = action_traj
            # print(time.time() - time0)
            

        # self.trajectories = trajectories

        return trajectories
    
    def generate_trajectories_batched(self, policy, num_traj, len_traj, reset_s0=False):
        # Ensure tensor on GPU with right dtype
        if not isinstance(policy, torch.Tensor):
            policy_t = torch.as_tensor(policy, dtype=torch.float32, device=self.device)
        elif policy.device != self.device or policy.dtype != torch.float32:
            policy_t = policy.to(self.device, dtype=torch.float32)
        else:
            policy_t = policy

        # Initialize starting states
        if reset_s0:
            s0 = torch.randint(0, self.mmdp.joint_transition_matrix_t.shape[0], (num_traj,), device=self.device)
        else:
            s0 = torch.as_tensor([self.mmdp.s0] * num_traj, dtype=torch.long, device=self.device)

        states = torch.zeros((num_traj, len_traj), dtype=torch.long, device=self.device)
        actions = torch.zeros((num_traj, len_traj), dtype=torch.long, device=self.device)

        state = s0
        for t in range(len_traj):
            # Get action probabilities for all current states
            action_probs = policy_t[state]  # shape: (num_traj, n_actions)
            action_probs /= action_probs.sum(dim=1, keepdim=True)

            # Sample actions for all trajectories in parallel
            action = torch.multinomial(action_probs, num_samples=1).squeeze(1)

            # Sample next states for all trajectories in parallel
            next_state_probs = self.mmdp.joint_transition_matrix_t[state, action, :]  # (num_traj, n_states)
            next_state = torch.multinomial(next_state_probs, num_samples=1).squeeze(1)

            states[:, t] = state
            actions[:, t] = action
            state = next_state

        return torch.stack((states, actions), dim=2)  # shape: (num_traj, len_traj, 2)

    
    
    def calculate_value_function(self, reward, policy, theta=1e-8):
        """
        Calculates value function with given reward and given policy
        
        reward: Reward matrix of MMDP. np.array with size(n_states, n_actions)
        policy: Stochastic policy of MMDP. np.array with size (n_states, n_actions)
        theta: Convergence threshold. float.
        
        return: Calculated value function. np.array with size (n_states,)
        """
        
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        gamma = self.mmdp.gamma
        transition_matrix = self.mmdp.transition_matrices
        
        reward = reward.reshape(n_states,n_actions)

        if n_states > 1000:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Assume transition_matrix, policy, and reward are already torch tensors on CPU
            transition_matrix = self.transition_matrix_torch.to(device)  # shape: (n_states, n_actions, n_states)
            if isinstance(policy, np.ndarray):
                policy = torch.tensor(policy, dtype=torch.float32, device=device) 
            else:
                policy = policy.to(device)                        # shape: (n_states, n_actions)
            if isinstance(reward, np.ndarray):
                reward = torch.tensor(reward, dtype=torch.float32, device=device) 
            else:
                reward = reward.to(device)                        # shape: (n_states, n_actions)
            gamma = self.mmdp.gamma

            # Compute P_pi
            P_pi = torch.zeros(n_states, n_states, device=device)
            for s1 in range(n_states):
                P_pi[s1] = torch.matmul(transition_matrix[s1].T, policy[s1])

            # Compute R_pi
            R_pi = torch.sum(policy * reward, dim=1)  # shape: (n_states,)

            # Compute value function
            I = torch.eye(n_states, device=device)
            value_function = torch.linalg.solve(I - gamma * P_pi, R_pi)  # shape: (n_states,)
            value_function = value_function.cpu().numpy()

        else:

            P_pi = np.zeros((n_states, n_states))  # State transition probability under stochastic policy
            for s1 in range(n_states):
                P_pi[s1,:] = np.dot(transition_matrix[s1, :, :].T, policy[s1,:]).T  #  ((n_states, n_actions) * (n_actions,))
                
            R_pi = np.sum(policy*reward, axis = 1) #(n_states, n_actions)*(n_actions, n_actions)
    
            value_function = np.linalg.inv(np.identity(n_states) - gamma*P_pi).dot(R_pi) #(n_states,)
            
        return value_function
    
    def estimate_value_function(self, trajectories):
        """
        Estimates value function using IRL based on given trajectories.
        
        trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
        
        return: Estimated value function. np.array with shape (n_states,)
        """
        
        n_states = self.mmdp.n_joint_states
        true_rewards = self.mmdp.joint_rewards

        time0 = time.time()
        
        print("Running IRL")
        estimated_rewards = self.irl(trajectories) # (n_states * n_actions, )
        # Deterministic Policy
        # estimated_policy  = np.argmax(value_iteration(expected_rewards.reshape(n_states,-1), n_states, n_actions, transition_probability)[1],
        #                                axis = -1) # (n_states, ) 
        # Stochastic Policy
        # estimated_policy  = self.value_iteration_cpu(estimated_rewards.reshape(n_states,-1))[1]
        # estimated_value_function = self.calculate_value_function(true_rewards, estimated_policy)

        print("Time :", time.time()-time0)

        # Update variable
        self.expected_rewards = estimated_rewards
        # self.estimated_policy = estimated_policy
        # self.estimated_value_function = estimated_value_function
        # self.self_estimated_value_function = self.calculate_value_function(self.expected_rewards, estimated_policy)
        
        return None

    def value_iteration_cpu(self, reward, theta=1e-8):
            """
            Runs the value iteration algorithm to find the optimal value function and policy from given reward.

            reward: Reward. np.array with shape (n_states, n_actions).
            theta: Convergence threshold. float.
            
            return: Tuple containing two np.arrays. The first array is of shape (n_states,) and represents the optimal
                    value function. The second array is of shape (n_states, n_actions) and represents the optimal policy.
            """
            n_states = self.mmdp.n_joint_states
            n_actions = self.mmdp.n_joint_actions
            gamma = self.mmdp.gamma
            transitions = self.mmdp.transition_matrices
            
            # Initialize the value function and policy arrays
            value_function = np.ones(n_states)
            policy = np.zeros((n_states, n_actions))
            
            
            if len(reward.shape) == 1:
                reward = reward.reshape(n_states, n_actions)
                # tf.reshape(reward, (n_states, n_actions))

            
            while True:
                delta = 0

                # Update the value function for each state
                for s in range(n_states):
                    v = value_function[s]
                    q_values = np.zeros(n_actions)
                    
                    # Compute Q(s,a) for all actions
                    for a in range(n_actions):
                        # Compute expected value of next state
                        next_state_probs = transitions[s][a] #(n_states)
                        expected_value = gamma * np.dot(next_state_probs, value_function)

                        # Add action-dependent reward to Q-value calculation
                        q_values[a] = reward[s,a] + expected_value
                        
                    q_values = q_values - np.max(q_values)
                    
                    value_function[s]= np.log(np.sum(np.exp(q_values)))
                    
                    policy[s] = np.exp(q_values)/np.exp(value_function[s])
                    
                    # Update delta
                    delta = max(delta, abs(v - value_function[s]))

                # Check for convergence
                if delta < theta:
                    break

            return value_function, policy
        

    def value_iteration_gpu_unvectorized(self, reward, theta=1e-5):

        if isinstance(reward, np.ndarray):
            reward = torch.tensor(reward, dtype=torch.float32, device=self.device)

        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        gamma = self.mmdp.gamma

        transitions = self.transition_matrix_torch

        value_function = torch.ones(n_states, dtype=torch.float32, device=self.device)  # (s,)
        policy = torch.zeros((n_states, n_actions), dtype=torch.float32, device=self.device)

        if reward.dim() == 1:
            reward = reward.view(n_states, n_actions)

        while True:
            delta = 0.0
            new_value_function = torch.empty_like(value_function)

            for s in range(n_states):
                v = value_function[s]
                q_values = torch.empty(n_actions, device=self.device)

                for a in range(n_actions):
                    next_state_probs = transitions[s, a]  # (s')
                    expected_value = gamma * torch.dot(next_state_probs, value_function)
                    q_values[a] = reward[s, a] + expected_value

                q_values = q_values - torch.max(q_values)  # for numerical stability
                new_v = torch.logsumexp(q_values, dim=0)
                new_value_function[s] = new_v
                policy[s] = torch.exp(q_values - new_v)  # softmax

                delta = max(delta, torch.abs(v - new_v).item())

            value_function = new_value_function

            if delta < theta:
                break

        return value_function, policy
    
    def value_iteration(self, reward, theta=1e-5):
        if isinstance(reward, np.ndarray):
            reward = torch.tensor(reward, dtype=torch.float32)
        reward = reward.to(self.device)

        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        gamma = self.mmdp.gamma

        transitions = self.transition_matrix_torch  # (s, a, s')
        transitions = transitions.to(self.device)

        if reward.dim() == 1:
            reward = reward.view(n_states, n_actions)

        value_function = torch.ones(n_states, dtype=torch.float32, device=self.device)

        while True:
            delta = 0.0

            # Compute Q(s, a) = r(s,a) + gamma * sum_s' T(s,a,s') * V(s')
            # Use einsum for fast batch matrix multiply:
            #   transitions: (s, a, s') * value_function: (s') -> (s, a)
            expected_values = gamma * torch.einsum('sas,s->sa', transitions, value_function)
            q_values = reward + expected_values  # shape (s, a)

            # Soft value iteration: V(s) = logsumexp(Q(s,a))
            max_q = torch.max(q_values, dim=1, keepdim=True)[0]
            logsumexp = max_q + torch.log(torch.sum(torch.exp(q_values - max_q), dim=1, keepdim=True))
            new_value_function = logsumexp.squeeze()

            # Policy: softmax(Q(s,a))
            policy = torch.exp(q_values - new_value_function.unsqueeze(1))  # shape (s, a)

            delta = torch.norm(value_function - new_value_function, p=float('inf')).item()
            value_function = new_value_function

            if delta < theta:
                break

        return value_function, policy


    def irl(self, trajectories):
        pass
    
    def find_svf(self, trajectories, discount = False):
        """
        Find the state visitation frequency from trajectories.

        n_states: Number of states. int.
        trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
        
        return: State visitation frequencies vector with shape (n_states,).
        """
        
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions

        # Convert to torch on device
        if not isinstance(trajectories, torch.Tensor):
            traj_t = torch.as_tensor(trajectories, dtype=torch.long, device=self.device)
        else:
            traj_t = trajectories.to(self.device).long()

        # Handle the shape (1, num_traj, len_traj, 2)
        if traj_t.dim() == 4:
            traj_t = traj_t[0]  # remove first dimension

        # Separate states and actions
        states = traj_t[:, :, 0]
        actions = traj_t[:, :, 1]
        
        num_traj, len_traj, _ = traj_t.shape
        
        if discount:
            discounts = self.mmdp.gamma ** torch.arange(len_traj, device=self.device)  # shape: (len_traj,)
            discounts = discounts.view(1, len_traj)  
            flat_indices = (states * n_actions + actions).view(-1)           # shape: (num_traj * len_traj,)
            weights = (discounts * torch.ones_like(states, dtype=torch.float32)).view(-1)  # same shape
            svf = torch.zeros(n_states * n_actions, device=self.device, dtype=torch.float32)
            svf = svf.index_add(0, flat_indices, weights)

        else:
            # Flatten to 1D
            flat_indices = states * n_actions + actions  # shape: (num_traj, len_traj)
            flat_indices = flat_indices.view(-1)

            # Count occurrences for each state-action pair
            svf = torch.bincount(flat_indices, minlength=n_states * n_actions).float()

        # Normalize by number of trajectories
        svf /= num_traj

        return svf
    
    def find_svf_cpu(self, trajectories):
        
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions

        if len(trajectories.shape) == 4:
            trajectories = trajectories[0,:,:,:]
            
        svf = np.zeros(n_states*n_actions)

        for trajectory in trajectories:
            for state, action in trajectory:
                svf[int(state*n_actions + action)] += 1

        svf /= trajectories.shape[0]

        return svf
    
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
        
        start_state_action_count = np.zeros(n_states * n_actions) # Initial Distribution
        for trajectory in trajectories: # (step, 2)
            state = int(trajectory[0, 0])
            action = int(trajectory[0, 1])
            start_state_action_count[state*n_actions + action] += 1
        p_start_state_action = start_state_action_count/n_trajectories

        expected_svf = np.tile(p_start_state_action, (trajectory_length, 1)).T # (n_states * n_action, step, 1)
        
        for t in range(1, trajectory_length):
            expected_svf[:, t] = 0 # (n_states * n_action, 1)
            for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
                expected_svf[i*n_actions + j, t] += (np.sum(expected_svf[k*n_actions:(k+1)*n_actions , t-1]) *
                                    policy[i, j] * transition_probability[i, j, k]) * discount

        return expected_svf.sum(axis=1) # (n_states * n_action, )
    
    def find_expected_svf(self, r, trajectories):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        gamma = self.mmdp.gamma
        transition_probability = self.transition_matrix_torch

        if len(trajectories.shape) == 4:
            trajectories = trajectories[0,:,:,:]

        n_trajectories = trajectories.shape[0]
        trajectory_length = trajectories.shape[1]

        # Reward to policy
        if not isinstance(r, torch.Tensor):
            r = torch.tensor(r, dtype=torch.float32, device=device).reshape(n_states, n_actions)
        _, policy = self.value_iteration(r, theta=1e-5)  # should return a (n_states, n_actions) PyTorch tensor
        policy = policy.to(device)

        # Initial state-action distribution
        # p_start = torch.zeros(n_states * n_actions, dtype=torch.float32, device=device)
        # for trajectory in trajectories:
        #     state = int(trajectory[0, 0])
        #     action = int(trajectory[0, 1])
        #     p_start[state * n_actions + action] += 1
        # p_start /= n_trajectories

        first_sa = trajectories[:, 0, :2].long()  # shape: (n_trajectories, 2)
        flat_indices = first_sa[:, 0] * n_actions + first_sa[:, 1]
        p_start = torch.zeros(n_states * n_actions, dtype=torch.float32, device=device)
        p_start.scatter_add_(0, flat_indices, torch.ones_like(flat_indices, dtype=torch.float32))
        p_start /= n_trajectories

        expected_svf = torch.zeros((n_states * n_actions, trajectory_length), dtype=torch.float32, device=device)
        expected_svf[:, 0] = p_start

        # Reshape transition matrix for batched computation: (s', a, s) -> (s', a, s) as is
        # But we need to compute for each t:
        # expected_svf[i * n_actions + j, t] += sum_over_k(expected_svf[k*a:(k+1)*a, t-1]) * policy[i, j] * T[i, j, k]

        # for t in range(1, trajectory_length):
        #     prev_svf = expected_svf[:, t-1].view(n_states, n_actions)
        #     marginal_prev_s = prev_svf.sum(dim=1)  # (n_states,)
            
        #     # Compute expected_svf at time t
        #     # For each s', a: sum over s of (marginal_prev_s[s] * T[s', a, s])
        #     weighted_transitions = transition_probability * marginal_prev_s.view(1, 1, -1)  # (s', a, s)
        #     marginal_next = weighted_transitions.sum(dim=2)  # (s', a)

        #     expected_svf[:, t] = (policy * marginal_next).view(-1) * gamma

        for t in range(1, trajectory_length):
            prev_svf = expected_svf[:, t-1].view(n_states, n_actions)   # (s, a)
            marginal_prev_s = prev_svf.sum(dim=1)                       # (s,)

            # Compute marginal_next[s', a] = ∑_s T[s', a, s] * marginal_prev_s[s]
            marginal_next = torch.einsum('sap,p->sa', transition_probability, marginal_prev_s)  # (s', a)

            # Update expected_svf for time t
            expected_svf[:, t] = (policy * marginal_next).reshape(-1) * gamma

        return expected_svf.sum(dim=1)  # (n_states * n_actions,)

  
    def softmax(x1, x2):
        """
        Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

        x1: float.
        x2: float.
        
        return: softmax(x1, x2)
        """
        max_x = max(x1, x2)
        min_x = min(x1, x2)
        return max_x + np.log(1 + np.exp(min_x - max_x))