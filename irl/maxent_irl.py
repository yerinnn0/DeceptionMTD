from .IRL import IRL
from itertools import product
from collections import defaultdict


import numpy as np
import numpy.random as rn
import scipy.sparse as sp
import time
import torch
from tqdm import tqdm


class MaxEntIRL(IRL):
    
    def __init__(self, mmdp, feature_map = "identity", epochs=5, learning_rate=0.9):
        super(MaxEntIRL, self).__init__(mmdp, feature_map)

        def sparse_scipy_to_torch(coo):
            values = torch.tensor(coo.data, dtype=torch.float32)
            indices = torch.vstack((torch.tensor(coo.row), torch.tensor(coo.col))).long()
            shape = coo.shape
            return torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float32)
    
        self.epochs = epochs
        self.learning_rate = learning_rate
        # self.feature_matrix = sp.coo_matrix(self.feature_matrix) # (n_states * n_actions, n_states * n_actions)
        self.feature_matrix = sparse_scipy_to_torch(self.feature_matrix).to(self.device)

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

        # coo = self.feature_matrix.tocoo()
        # indices = torch.tensor(np.array([coo.row, coo.col]), dtype=torch.int64, device=self.device)
        # values = torch.tensor(coo.data, dtype=torch.float32, device=self.device)
        # shape = coo.shape

        # feature_matrix = torch.sparse_coo_tensor(indices, values, shape, device=self.device)

        feature_matrix = self.feature_matrix.to(self.device)
        # feature_matrix = torch.tensor(self.feature_matrix, dtype=torch.float32, device=self.device)  # (n_states * n_actions, n_features)
        feature_expectations = self.find_feature_expectations(trajectories)
        feature_expectations = feature_expectations.to(dtype=torch.float32, device=self.device)
        trajectories_torch = torch.tensor(trajectories, dtype = torch.float32, device = self.device)
    
        alpha = torch.randn(feature_matrix.shape[1], dtype=torch.float32, device=self.device) #(n_features,)

        for i in tqdm(range(self.epochs)):
            # time0 = time.time()
            r = feature_matrix @ alpha  # (n_states * n_actions,)
            expected_svf = self.find_expected_svf(r, trajectories_torch)  # (n_states * n_actions,)
            
            grad = feature_expectations - feature_matrix.T @ expected_svf
            alpha += self.learning_rate * grad

            # print("epoch:", i, "Grad:", grad.T@grad, "Time: ", time.time()-time0)

        return (feature_matrix @ alpha).cpu().numpy().reshape(-1,)  # (n_states * n_actions, )

    def find_feature_expectations(self, trajectories):
        """
        Find the feature expectations for the given trajectories. This is the average path feature vector.

        feature_matrix: Matrix with the nth row representing the nth state. np.array with shape (n_states*n_action, 1)
        trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
        
        return: Feature expectations vector with shape (n_states * n_actions,).
        """
        n_actions = self.mmdp.n_joint_actions

        feature_expectations = torch.zeros(self.feature_matrix.shape[1], device=self.device) # (n_states * n_actions,)

        if len(trajectories.shape) == 4:
            trajectories = trajectories[0,:,:,:] # trajectories : (traj_number, steps, 2)


        fm = self.feature_matrix.coalesce().to(self.device)
        indices = fm.indices()   # [2, nnz]
        values  = fm.values()    # [nnz]
        num_rows, feature_dim = fm.shape

        ######################################################################
        # PART 1 — build CSR (GPU)
        ######################################################################
        row_idx = indices[0]        # [nnz]
        col_idx = indices[1]        # [nnz]

        # count nnz per row
        nnz_per_row = torch.bincount(row_idx, minlength=num_rows)

        # CSR row pointer
        row_ptr = torch.zeros(num_rows + 1, device=self.device, dtype=torch.long)
        row_ptr[1:] = torch.cumsum(nnz_per_row, dim=0)

        ######################################################################
        # PART 2 — flatten all trajectories into one row-index tensor
        ######################################################################
        row_list = []
        for traj in trajectories:
            traj_t = torch.as_tensor(traj, device=self.device)
            s = traj_t[:, 0].long()
            a = traj_t[:, 1].long()
            row_list.append(s * n_actions + a)

        all_rows = torch.cat(row_list, dim=0)   # [T_total]

        ######################################################################
        # PART 3 — FULLY VECTORIZED Sparse Row Gather → Dense Scatter
        ######################################################################
        # Step A: compute how many total nonzeros we will gather
        counts = nnz_per_row[all_rows]               # [T_total]
        total = counts.sum()

        # Step B: compute destination offsets for scatter
        offsets = torch.zeros_like(all_rows)
        offsets[1:] = torch.cumsum(counts, dim=0)[:-1]

        # Step C: build full index list
        gather_indices = torch.arange(total, device=self.device)
        # map gather_indices → original sparse nnz indices
        # invert CSR
        # For each row in all_rows, we add all col_idx[row_ptr[r]:row_ptr[r+1]]

        # Build mapping: for each r in all_rows, expand the range row_ptr[r]:row_ptr[r+1]
        expanded = torch.arange(total, device=self.device)  # just a placeholder for shape
        # actual map
        src_indices = torch.empty(total, dtype=torch.long, device=self.device)

        # Uses CSR indexing trick: we iterate over rows in one batched kernel
        ptr_starts = row_ptr[all_rows]
        ptr_ends   = row_ptr[all_rows + 1]

        # Compute "repeated ranges" in one pass
        diff = ptr_ends - ptr_starts
        repeats = torch.repeat_interleave(diff)
        src_indices = torch.cat([ torch.arange(ptr_starts[i], ptr_ends[i], device=self.device) 
                                for i in range(all_rows.size(0)) ])

        # src_indices now maps each merged gather index → original value index in "values"

        # Step D: gather sparse values and columns
        gathered_cols  = col_idx[src_indices]   # [total]
        gathered_vals  = values[src_indices]    # [total]

        ######################################################################
        # PART 4 — scatter_add to dense feature expectations
        ######################################################################
        feature_expectations = torch.zeros(feature_dim, device=self.device)
        feature_expectations.scatter_add_(0, gathered_cols, gathered_vals)

                # row_index = int(state * n_actions + action)
                # # When self.feature_matrix is a Tensor
                # feature_expectations += self.feature_matrix[row_index]
                # # Extract the feature vector using the COO format
                # indices = self.feature_matrix.row == row_index
                # feature_vector = np.zeros(self.feature_matrix.shape[1])
                # feature_vector[self.feature_matrix.col[indices]] = self.feature_matrix.data[indices]
                # feature_expectations += feature_vector

                # feature_expectations += feature_matrix[int(state * n_actions + action)]

        feature_expectations /= trajectories.shape[0]

        return feature_expectations






