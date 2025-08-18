from .IRL import IRL
from itertools import product


import numpy as np
import numpy.random as rn
import scipy.sparse as sp
import time
import torch


class MaxEntIRL(IRL):
    
    def __init__(self, mmdp, feature_map = "identity", epochs=5, learning_rate=0.9):
        super(MaxEntIRL, self).__init__(mmdp, feature_map)
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.feature_matrix = sp.coo_matrix(self.feature_matrix) # (n_states * n_actions, n_states * n_actions)


    def irl_cpu(self, trajectories):
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

        n_states_actions, d_states_actions = self.feature_matrix.shape # n_states * n_actions
        n_states = int(self.feature_matrix.shape[0]/n_actions)

        # Initialize weights.
        alpha = rn.uniform(size=(d_states_actions,)) # Reward: (n_states * n_actions,)

        # Calculate the feature expectations \tilde{phi}.
        feature_expectations = self.find_feature_expectations(trajectories) # (n_states * n_actions, )
        
        # trajectories : (traj_number, steps, 2)

        # Gradient descent on alpha.
        for i in range(self.epochs):
            print("epoch:", i)
            r = self.feature_matrix.dot(alpha)
            expected_svf = self.find_expected_svf(r, trajectories)  # (n_states * n_actions, )
            grad = feature_expectations - self.feature_matrix.T.dot(expected_svf)   # (n_states * n_actions, )

            alpha += self.learning_rate * grad   # (n_states * n_actions, )

        return self.feature_matrix.dot(alpha).reshape((n_states * n_actions,))  # (n_states * n_actions, )
    
    def irl(self, trajectories):

        coo = self.feature_matrix.tocoo()
        indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.int64, device=self.device)
        values = torch.tensor(coo.data, dtype=torch.float32, device=self.device)
        shape = coo.shape

        feature_matrix = torch.sparse_coo_tensor(indices, values, shape, device=self.device)

        # feature_matrix = torch.tensor(self.feature_matrix, dtype=torch.float32, device=device)  # (n_states * n_actions, n_features)
        feature_expectations = torch.tensor(self.find_feature_expectations(trajectories), dtype=torch.float32, device=self.device)
        trajectories_torch = torch.tensor(trajectories, dtype = torch.float32, device = self.device)
        
        alpha = torch.randn(feature_matrix.shape[1], dtype=torch.float32, device=self.device) #(n_features,)

        for i in range(self.epochs):
            time0 = time.time()
            r = feature_matrix @ alpha  # (n_states * n_actions,)
            expected_svf = self.find_expected_svf(r, trajectories_torch)  # (n_states * n_actions,)
            
            grad = feature_expectations - feature_matrix.T @ expected_svf
            alpha += self.learning_rate * grad

            print("epoch:", i, "Grad:", grad.T@grad, "Time: ", time.time()-time0)

        return (feature_matrix @ alpha).cpu().numpy().reshape(-1,)  # (n_states * n_actions, )

    def find_feature_expectations(self, trajectories):
        """
        Find the feature expectations for the given trajectories. This is the average path feature vector.

        feature_matrix: Matrix with the nth row representing the nth state. np.array with shape (n_states*n_action, 1)
        trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
        
        return: Feature expectations vector with shape (n_states * n_actions,).
        """
        n_actions = self.mmdp.n_joint_actions

        feature_expectations = np.zeros(self.feature_matrix.shape[1]) # (n_states * n_actions,)

        if len(trajectories.shape) == 4:
            trajectories = trajectories[0,:,:,:] # trajectories : (traj_number, steps, 2)

        for trajectory in trajectories: # (steps, 2)
            #for state, action in trajectory:
            for step in trajectory:
                state = step[0]
                action = step[1]

                row_index = int(state * n_actions + action)
                # Extract the feature vector using the COO format
                indices = self.feature_matrix.row == row_index
                feature_vector = np.zeros(self.feature_matrix.shape[1])
                feature_vector[self.feature_matrix.col[indices]] = self.feature_matrix.data[indices]
                feature_expectations += feature_vector

                # feature_expectations += feature_matrix[int(state * n_actions + action)]

        feature_expectations /= trajectories.shape[0]

        return feature_expectations






