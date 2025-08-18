
from .IRL import IRL

import numpy as np
import torch

class ApprenticeshipIRL(IRL):
    
    def __init__(self, mmdp, feature_map = "identity", threshold = 1, max_iter = 50):
        super(ApprenticeshipIRL, self).__init__(mmdp, feature_map)
        
        
        def sparse_scipy_to_torch(coo):
            values = torch.tensor(coo.data, dtype=torch.float32)
            indices = torch.vstack((torch.tensor(coo.row), torch.tensor(coo.col))).long()
            shape = coo.shape
            return torch.sparse_coo_tensor(indices, values, size=shape, dtype=torch.float32)
        
        self.threshold = threshold
        self.max_iter = max_iter
        self.feature_matrix_t = sparse_scipy_to_torch(self.feature_matrix).to(self.device)
        
    def irl(self, trajectories):
        
        estimated_rewards = self.apprenticeship_algorithm(trajectories, self.threshold, self.max_iter)
        
        return estimated_rewards
    
    def apprenticeship_algorithm(self, trajectories, threshold = 1, max_iter = 50):
        
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        reward_scale_factor = 1
                    
        # ---------- Helper: convert policy numpy -> torch (if needed) ----------
        def to_torch_policy(np_policy):
            # expects np_policy shaped (n_states, n_actions)
            return torch.as_tensor(np_policy, dtype=torch.float32, device=self.device)

        def to_numpy_policy(t_policy):
            # t_policy is torch tensor on device
            return t_policy.detach().cpu().numpy()

        # ---------- expert_svf ----------
        # Preferred: your find_svf should return a torch tensor on device.
        expert_svf_t = None
        expert_svf_res = self.find_svf(trajectories, discount = True)
        if isinstance(expert_svf_res, torch.Tensor):
            expert_svf_t = expert_svf_res.to(self.device).float()
        else:
            expert_svf_t = torch.as_tensor(expert_svf_res, dtype=torch.float32, device=self.device)

        # ---------- initial policy ----------
        policy_np = np.ones((n_states, n_actions), dtype=np.float32) / float(n_actions)
        policy_t = to_torch_policy(policy_np)  # shape (n_states, n_actions)

        # ---------- mu, mu_bar, w, t ----------
        # Preferred: find_feature_estimation_from_policy accepts torch policy and returns torch
        mu_res = self.find_feature_estimation_from_policy(policy_t)
        if isinstance(mu_res, torch.Tensor):
            mu = mu_res.to(self.device).float()
        else:
            mu = torch.as_tensor(mu_res, dtype=torch.float32, device=self.device)

        mu_bar = mu.clone()
        w = expert_svf_t - mu_bar
        t_norm = torch.norm(w, p=2).item()  # use .item() for scalar threshold checks

        # compute estimated_rewards = feature_matrix.dot(w) (w is feature-dim vector)
        # ensure w has correct shape: (n_features,) or (n_features,1)
        # feature_matrix shape: (n_states*n_actions, n_features)
        estimated_rewards_t = self.feature_matrix_t.matmul(w).reshape(n_states * n_actions) *reward_scale_factor  # torch tensor

        # ---------- value iteration ----------
        # Preferred: value_iteration accepts torch rewards and returns torch policy on device
        # If your value_iteration returns numpy, we'll transfer it back to torch below.
        value_res = self.value_iteration(estimated_rewards_t, theta=1e-4)
        # handle different return types/structures:
        if isinstance(value_res, tuple):
            _, policy_out = value_res
        else:
            # if it returns only policy
            policy_out = value_res

        if isinstance(policy_out, torch.Tensor):
            policy_t = policy_out.reshape((n_states, n_actions)).to(self.device).float()
        else:
            # assume numpy; copy to torch device
            policy_t = to_torch_policy(np.asarray(policy_out).reshape((n_states, n_actions)))

        iter_count = 1

        # ---------- main loop ----------
        while (t_norm > threshold) and (iter_count < max_iter):
            # compute mu for current policy; prefer GPU helper that takes torch
            mu_res = self.find_feature_estimation_from_policy(policy_t)
            if isinstance(mu_res, torch.Tensor):
                mu = mu_res.to(self.device).float()
            else:
                mu = torch.as_tensor(mu_res, dtype=torch.float32, device=self.device)

            direction = mu - mu_bar  # torch
            # safe dot product: both vectors should be 1-D of same shape
            denom = torch.dot(direction, direction)
            if denom.abs().item() < 1e-12:
                break
            alpha = torch.dot(direction, w) / denom
            alpha_item = alpha.item()

            if abs(alpha_item) < 1e-8:
                break

            step_size = 0.5/ np.sqrt(iter_count/10 + 1)
            # update mu_bar (torch operations on GPU)
            mu_bar = mu_bar + (step_size * alpha) * direction

            w = expert_svf_t - mu_bar
            t_norm = torch.norm(w, p=2).item()

            # update rewards and policy
            estimated_rewards_t = self.feature_matrix_t.matmul(w).reshape(n_states * n_actions) *reward_scale_factor

            value_res = self.value_iteration(estimated_rewards_t, theta=1e-4)
            if isinstance(value_res, tuple):
                _, policy_out = value_res
            else:
                policy_out = value_res

            if isinstance(policy_out, torch.Tensor):
                policy_t = policy_out.reshape((n_states, n_actions)).to(self.device).float()
            else:
                policy_t = to_torch_policy(np.asarray(policy_out).reshape((n_states, n_actions)))

            iter_count += 1
            if iter_count % 10 == 0:
                print(iter_count, t_norm)

        # return estimated_rewards as numpy vector (to match original signature), but keep GPU option
        return estimated_rewards_t.detach().cpu().numpy()

        
        # expert_svf = self.find_svf(trajectories)
        
        # policy =  np.ones((n_states, n_actions)) / n_actions
        # mu = self.find_feature_estimation_from_policy(policy)
        # mu_bar = mu
        # w = expert_svf - mu_bar
        # t = np.sqrt(w.T.dot(w))
        
        # estimated_rewards = (self.feature_matrix.dot(w)).reshape((n_states * n_actions,)) 
        # _, policy = self.value_iteration(estimated_rewards)
        # policy = policy.reshape((n_states, n_actions)) 
        # iter = 1
                
        # while (t > threshold) and (iter < max_iter):
        #     mu = self.find_feature_estimation_from_policy(policy)
        #     direction = mu - mu_bar
        #     alpha = direction.T.dot(w)/direction.T.dot(direction)

            
        #     if abs(alpha) < 1e-8:
        #         break

        #     step_size = 0.5 / np.sqrt(iter/10+1)
        #     momentum = 0.2
            
        #     # if abs(alpha) < 1e-5:
        #     #     update_dir = momentum * update_dir + (1 - momentum) * direction
        #     #     mu_bar += step_size * update_dir
        #     # else:
        #     #     mu_bar += step_size* alpha * direction
        #     # update_dir = direction 

        #     mu_bar += step_size* alpha * direction
            
        #     w = expert_svf - mu_bar
        #     t = np.linalg.norm(w, 2)
        #     estimated_rewards = (self.feature_matrix.dot(w)).reshape((n_states * n_actions,)) 
        #     _, policy = self.value_iteration(estimated_rewards)
        #     policy = policy.reshape((n_states, n_actions)) 
            
        #     iter += 1
        #     if iter % 10 == 0:
        #         print(iter, t)
        
        # return estimated_rewards
    
    def find_feature_estimation_from_trajectory(self, trajectories):
        
        # Discounted
        svf = self.find_svf(trajectories, discount = True)
        estimated_value = (self.feature_matrix_t)@(svf)
        
        return estimated_value

    def find_feature_estimation_from_policy(self, policy):
        
        # Make sure policy tensor is on device
        if not isinstance(policy, torch.Tensor):
            policy_t = torch.as_tensor(policy, dtype=torch.float32, device=self.device)
        else:
            if policy.device != self.device or policy.dtype != torch.float32:
                policy_t = policy.to(device=self.device, dtype=torch.float32)
            else:
                policy_t = policy
        
        trajectories = self.generate_trajectories_batched(policy_t, num_traj =100, len_traj = 100)
        estimated_value = self.find_feature_estimation_from_trajectory(trajectories)
        
        return estimated_value