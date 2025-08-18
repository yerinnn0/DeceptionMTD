"""
Solve optimization problem for deceptive policy
"""
import numpy as np
import time
import scipy.sparse as sp

import torch


class PolicyOptimization:

    def __init__(self, mmdp):
        self.mmdp = mmdp
        
        self.n_states = self.mmdp.n_joint_states
        self.n_actions = self.mmdp.n_joint_actions
        self.gamma = self.mmdp.gamma
        self.transitions = self.mmdp.transition_matrices
        self.r = self.mmdp.joint_rewards
        
        ##########################################################
        
        time0 = time.time()

        self.A_fl = sp.lil_matrix((self.n_states, self.n_states * self.n_actions))
        T = self.transitions.reshape(-1,self.n_states).T
        for s in range(self.n_states):
            self.A_fl[s, s * self.n_actions:(s + 1) * self.n_actions] = 1
            # for a in range(n_actions):
            #     for s2 in range(n_states):
            #         assert transitions[s,a,s2] == T[s2, s * n_actions + a]
            #         self.A_fl[s2, s * n_actions + a] -= gamma * transitions[s, a, s2]
        self.A_fl -= self.gamma*T
        self.A_fl = sp.csr_matrix(self.A_fl)
        
        self.A_r = sp.csr_matrix(-np.sum(self.transitions[:, :, self.mmdp.goal_states], axis=-1).reshape(1, -1))
        
        self.A_p = -2 * sp.eye(self.n_states * self.n_actions, format='csr')
        
        self.A_eq = sp.lil_matrix((self.n_states, self.n_actions),dtype = float)
        for s in range(self.n_states):
            if s in self.mmdp.goal_states:
                self. A_eq[s, :] = self.A_eq[s, :].toarray().flatten() + 1 
            if s in self.mmdp.decoy_states:
                self. A_eq[s, :] = self.A_eq[s, :].toarray().flatten() - 1 
        self.A_eq = 1 * self.A_eq.reshape(1, -1).tocsr()
        
        self.b_fl = np.array(self.mmdp.initial_joint_distribution, dtype=np.float64).reshape(-1, 1)
        self.b_r = np.array([-self.mmdp.v_reach], dtype=np.float64).reshape(-1, 1)
        self.b_p = np.zeros((self.n_states * self.n_actions, 1), dtype=np.float64)
        self.b_eq = np.array([0], dtype=np.float64).reshape(-1, 1)
        
        self.c_MDP = -self.r.flatten()
        # Inequality constraint: Gx <= h
        self.G = sp.vstack([self.A_p, self.A_r])
        self.h = np.vstack([self.b_p, self.b_r]).flatten()
        # Equality constraint: Ax = b
        self.A = self.A_fl
        self.b = self.b_fl.flatten()
        
                        
        print("Time for setting up optimization matrices:", time.time()-time0)
        ##########################################################

    # def scipy_to_torch_sparse(self, scipy_mat, device='cuda'):
    #     scipy_coo = scipy_mat.tocoo()
    #     indices = torch.tensor([scipy_coo.row, scipy_coo.col], dtype=torch.long, device=device)
    #     values = torch.tensor(scipy_coo.data, dtype=torch.float32, device=device)
    #     return torch.sparse_coo_tensor(indices, values, scipy_mat.shape, device=device)

    # def flow_constraint_matrix(self, X_flat):

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     self.A_fl_torch = self.scipy_to_torch_sparse(self.A_fl, device)
    #     self.b_fl_torch = torch.tensor(self.b_fl, dtype=torch.float32, device=device)

    #     return torch.sparse.mm(self.A_fl_torch, X_flat.unsqueeze(1)).squeeze() - self.b_fl_torch

    # def reachability_constraint_matrix(self, X_flat):
        
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #     self.A_r_torch = self.scipy_to_torch_sparse(self.A_r, device)
    #     self.b_r_torch = torch.tensor(self.b_r, dtype=torch.float32, device=device)

    #     return torch.sparse.mm(self.A_r_torch, X_flat.unsqueeze(1)).squeeze() - self.b_r_torch
        
    def flow_constraint_matrix(self,X_flat):
    
        return self.A_fl@X_flat - self.b_fl.flatten()

    def reachability_constraint_matrix(self,X_flat):
        
        return self.A_r@X_flat - self.b_r.flatten()
    
    def equivocal_constraint_matrix(self, X_flat):

        return self.A_eq@X_flat - self.b_eq.flatten()
    
    def flow_constraint(self,X_flat):
        
        X = X_flat.reshape((self.mmdp.n_joint_states, self.mmdp.n_joint_actions))
        occupancy_out = np.sum(X, axis=1)
        occupancy_in = np.zeros(self.mmdp.n_joint_states)
        for s2 in range(self.mmdp.n_joint_states):
            for s in range(self.mmdp.n_joint_states):
                for a in range(self.mmdp.n_joint_actions):
                    occupancy_in[s2] += self.mmdp.transition_matrices[s, a, s2] * X[s, a]
                    
        return occupancy_out - self.mmdp.gamma * occupancy_in - self.mmdp.initial_distribution

    def reachability_constraint(self,X_flat):

        X = X_flat.reshape((self.mmdp.n_joint_states, self.mmdp.n_joint_actions)) #(n_states, n_actions)
        # X_scaled = X/np.sum(X)
        transitions = self.mmdp.transition_matrices # (n_states, n_actions, n_states)

        P = sum([X * transitions[:,:,i] for i in self.mmdp.goal_states])
        P = np.sum(P)

        return P - self.mmdp.v_reach 
    
    def equivocal_constraint(self, X_flat):

        X = X_flat.reshape(self.mmdp.n_joint_states, self.mmdp.n_joint_actions)
    
        return np.sum(X[self.mmdp.goal_states]) - np.sum(X[self.mmdp.decoy_states])
    
        
    def solve_MDP(self):
        
        """
        Solve MMDP without deception (Optimization Problem 3)
        """
        pass
    
    def diversionary_deception(self, original_occupancy_measures, beta = 1):
        """
        Solve MMDP with diversionary deception (Optimization Problem 4)
        """
        pass
    
    def targeted_deception(self, target_occupancy_measures, beta = 1):
        """
        Solve MMDP with targeted deception (Optimization Problem 5)
        """
        pass
        
    def equivocal_deception(self):
        """
        Solve MMDP with equivocal deception (Optimization Problem 6)
        """
        pass
    
    def policy_evaluation(self,transitions, rewards, policy, eta, gamma):
        """
        Calculates a value function from given transitions, rewards, policy, eta, and gamma
        
        transitions: Transition probability matrix of MMDP. np.array with shape (n_joint_states, n_joint_actions, n_joint_states).
        rewards: Reward matrix of MMDP. np.array with shape (n_joint_states, n_joint_actions).
        policy: Policy of MMDP. np.array with shape (n_joint_states, n_joint_actions).
        eta: Stopping criteria. float.
        gamma: Discount factor of MMDP. float.
        return: Calculated value function. np.array with shape (n_joint_states,).
        """
        n_joint_states, n_joint_actions, _ = transitions.shape
        value_function = np.zeros(n_joint_states)
        delta = float('inf')
        
        while delta > eta:
            delta = 0
            for s in range(n_joint_states):
                v = value_function[s]
                new_v = 0
                for a in range(n_joint_actions):
                    prob_a = policy[s, a]
                    expected_reward = rewards[s, a]
                    for s2 in range(n_joint_states):
                        prob = transitions[s, a, s2]
                        expected_reward += gamma * prob * value_function[s2]
                    new_v += prob_a * expected_reward
                value_function[s] = new_v
                delta = max(delta, abs(v - value_function[s]))
             
        return value_function
    
    
    def evaluation(self, occupancy_measure):

        time0 = time.time()
        
        assert occupancy_measure.shape[0] == self.n_states*self.n_actions
        
        # print("Flow Constraint Error:", self.flow_constraint(occupancy_measure) - self.flow_constraint_matrix(occupancy_measure))
        # print("Task Constraint Error:", self.reachability_constraint(occupancy_measure) + self.reachability_constraint_matrix(occupancy_measure))
        # print("Equality Constraint Error:", self.equivocal_constraint(occupancy_measure) - self.equivocal_constraint_matrix(occupancy_measure))
        
        # feasibility check
        if (occupancy_measure<-1e-2).any():
            print("***********NON-NEGATIVITY VIOLATED*************")
            print(np.min(occupancy_measure))
        occupancy_measure = np.where((occupancy_measure >= 1e-8), occupancy_measure, 0)
        if (self.flow_constraint(occupancy_measure) >= 1e-2).any() or (self.flow_constraint(occupancy_measure) <= -1e-2).any():
            print("***********FLOW CONSTRAINT VIOLATED*************")
            print(np.max(np.abs(self.flow_constraint(occupancy_measure))))
        if (self.reachability_constraint(occupancy_measure) < -1e-4):
            print("***********TASK CONSTRAINT VIOLATED*************")
            print(np.max(self.reachability_constraint(occupancy_measure)))
        # if (self.equivocal_constraint(occupancy_measure) >= 1e-4).any():
        #     print("***********SOLUTION NOT FEASIBLE*************")
        occupancy_measures = occupancy_measure.reshape((self.n_states, self.n_actions))
        occupancy_measures = np.where(occupancy_measures >= 1e-9, occupancy_measures, 1e-9)
        denominator = np.repeat(np.sum(occupancy_measures, axis = -1).reshape(self.n_states,1), self.n_actions, axis = -1)
        denominator = np.where(denominator >= 1e-9, denominator, 1e-9)
        policy = occupancy_measures / denominator
        value_function = self.policy_evaluation(self.transitions, self.r, policy, 1e-6, self.gamma)
        revenue = np.sum(self.r * occupancy_measures)
        
        print("Time for evaluating solution:", time.time()-time0)

        return occupancy_measures, policy, value_function, revenue
    
