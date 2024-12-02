"""
Solve optimization problem for deceptive policy
"""
import numpy as np
from scipy.optimize import minimize
import time
from cvxopt import matrix, solvers

def policy_evaluation(transitions, rewards, policy, eta, gamma):
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


class PolicyOptimizationCVX:

    def __init__(self, mmdp):
        self.mmdp = mmdp

        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        gamma = self.mmdp.gamma
        transitions = self.mmdp.transition_matrices
        r = self.mmdp.joint_rewards
        
        self.threshold = 1e-4
        
        solvers.options['show_progress'] = False
        
        # Flow constraint
        self.A_fl = np.zeros((n_states, n_states * n_actions))
        self.b_fl = self.mmdp.initial_distribution
        for s in range(n_states):
            self.A_fl[s, s * n_actions:(s + 1) * n_actions] = 1
            for a in range(n_actions):
                for s2 in range(n_states):
                    self.A_fl[s2, s * n_actions + a] -= gamma * transitions[s, a, s2]
            
        # Reachability constraint (Task constraint)        
        self.A_r = -np.sum(transitions[:,:,self.mmdp.goal_states], axis = -1).reshape(1,-1)
        self.b_r = -self.mmdp.v_reach
        
        # Non-negativity constraint
        self.A_p = -200 * np.eye(n_states * n_actions)
        self.b_p = np.zeros((n_states * n_actions,1))
        
        # Equivocal constraint
        self.A_eq = np.zeros((n_states, n_actions))
        for s in range(n_states):
            for a in range(n_actions):
                if s in self.mmdp.goal_states:
                    self.A_eq[s,:] += 1
                if s in self.mmdp.decoy_states:
                    self.A_eq[s,:] -= 1
        self.A_eq = 100*self.A_eq.reshape(1,-1)
        self.b_eq = 0

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
        X_scaled = X
        transitions = self.mmdp.transition_matrices # (n_states, n_actions, n_states)

        P = sum([X_scaled * transitions[:,:,i] for i in self.mmdp.goal_states])
        P = np.sum(P)

        return P - self.mmdp.v_reach 
    
    def equivocal_constraint(self, X_flat):

        X = X_flat.reshape(self.mmdp.n_joint_states, self.mmdp.n_joint_actions)
    
        return np.sum(X[self.mmdp.goal_states]) - np.sum(X[self.mmdp.decoy_states])

    # def solve_linear_programming(self):
    #     """
    #     Solve MMDP without deception (Optimization Problem 3)
    #     """

    #     n_states = self.mmdp.n_joint_states
    #     n_actions = self.mmdp.n_joint_actions
    #     gamma = self.mmdp.gamma
    #     transitions = self.mmdp.transition_matrices
    #     r = self.mmdp.joint_rewards

    #     def objective_LP(X_flat):
    #         X = X_flat.reshape((n_states, n_actions))
    #         return -np.sum(r * X)

    #     initial_guess = np.ones(n_states*n_actions)
    #     constraints = [{'type': 'eq', 'fun': self.flow_constraint},{'type': 'ineq', 'fun': self.reachability_constraint}]

    #     print("Solving LP...")
    #     time0 = time.time()

    #     result = minimize(objective_LP, initial_guess, constraints=constraints, bounds=[(0, None) for _ in range(n_states * n_actions)], options={'disp': False})

    #     print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

    #     sol = result.x
    #     sol = np.where(sol >= 1e-8, sol, 0)
    #     # feasibility check
    #     if (self.flow_constraint(sol) >= 1e-6).all():
    #         print("***********SOLUTION NOT FEASIBLE*************")
    #     occupancy_measures = sol.reshape((n_states, n_actions))
    #     denominator = np.repeat(np.sum(occupancy_measures, axis = -1).reshape(n_states,1), n_actions, axis = -1)
    #     denominator = np.where(denominator >= 1e-9, denominator, 1e-9)
    #     policy = occupancy_measures / denominator
    #     value_function = policy_evaluation(transitions, r, policy, 1e-6, gamma)
    #     revenue = np.sum(r * occupancy_measures)

    #     return occupancy_measures, policy, value_function, revenue
    
    def solve_lp_cvxopt(self):
        """
        Solve MMDP without deception (Optimization Problem 3)
        """
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        gamma = self.mmdp.gamma
        transitions = self.mmdp.transition_matrices
        r = self.mmdp.joint_rewards

        print("Solving LP...")
        time0 = time.time()
        
        c = -r.flatten()
        # Inequality constraint: Gx <= h
        G = np.vstack([self.A_p, self.A_r])
        h = np.vstack([self.b_p, self.b_r])
        # Equality constraint: Ax = b
        A = np.vstack([self.A_fl])
        b = np.hstack([self.b_fl])

        # Convert the problem into cvxopt format
        G_cvxopt = matrix(G)
        h_cvxopt = matrix(h)
        A_cvxopt = matrix(A)
        b_cvxopt = matrix(b)
        c_cvxopt = matrix(c)
        
        sol_dict = solvers.lp(c_cvxopt, G_cvxopt, h_cvxopt, A_cvxopt, b_cvxopt)
        
        sol = np.array(sol_dict['x']).flatten()
        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        sol = np.where(sol >= 1e-8, sol, 0)
        # feasibility check
        if (self.flow_constraint(sol) >= 1e-6).all():
            print("***********SOLUTION NOT FEASIBLE*************")
        occupancy_measures = sol.reshape((n_states, n_actions))
        occupancy_measures = np.where((occupancy_measures < 0) & (occupancy_measures > self.threshold), 0, occupancy_measures)
        denominator = np.repeat(np.sum(occupancy_measures, axis = -1).reshape(n_states,1), n_actions, axis = -1)
        denominator = np.where(denominator >= 1e-9, denominator, 1e-9)
        policy = occupancy_measures / denominator
        value_function = policy_evaluation(transitions, r, policy, 1e-6, gamma)
        revenue = np.sum(r * occupancy_measures)

        return occupancy_measures, policy, value_function, revenue
    
    def diversionary_deception(self, occupancy_measures, beta = 0):
        """
        Solve MMDP with diversionary deception (Optimization Problem 4)
        """
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        beta = beta
        gamma = self.mmdp.gamma
        transitions = self.mmdp.transition_matrices
        r = self.mmdp.joint_rewards

        def objective_QP(X_flat):
            X = X_flat.reshape((n_states, n_actions))
            return -np.sum(beta * X**2 + (r - 2 * beta * occupancy_measures) * X)
        
        initial_guess = np.ones(n_states*n_actions)
        
        constraints = [{'type': 'eq', 'fun': self.flow_constraint}, {'type': 'ineq', 'fun': self.reachability_constraint}]

        print("Solving QP...")
        
        time0 = time.time()
        result = minimize(objective_QP, initial_guess, constraints=constraints, bounds=[(0, None) for _ in range(n_states * n_actions)], options={'disp': False})

        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        sol = result.x
        # feasibility check
        if (self.flow_constraint(sol) >= 1e-6).all():
            print("***********SOLUTION NOT FEASIBLE*************")
        deceptive_occupancy_measures = sol.reshape((n_states, n_actions))
        denominator = np.repeat(np.sum(deceptive_occupancy_measures, axis = -1).reshape(n_states,1), n_actions, axis = -1)
        denominator = np.where(denominator >= 1e-9, denominator, 1e-9)
        deceptive_policy = deceptive_occupancy_measures / denominator
        deceptive_value_function = policy_evaluation(transitions, r, deceptive_policy, 1e-6, gamma)
        deceptive_revenue = np.sum(r * deceptive_occupancy_measures)

        return deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue
    
        # """
        # Solve MMDP with diversionary deception (Optimization Problem 4)
        # """
        # n_states = self.mmdp.n_joint_states
        # n_actions = self.mmdp.n_joint_actions
        # beta = beta
        # gamma = self.mmdp.gamma
        # transitions = self.mmdp.transition_matrices
        # r = self.mmdp.joint_rewards

        # print("Solving Diversionary Deception...")
        # time0 = time.time()
        
        # P = - (2 * beta) * np.eye(n_states * n_actions)
        # q = (2 * beta) * occupancy_measures.flatten() - r.flatten()

        # # Objective function
        # P = matrix(P)
        # q = matrix(q)
        # # Inequality constraint: Gx <= h
        # G = matrix(np.vstack([self.A_p, self.A_r]))
        # h =  matrix(np.vstack([self.b_p, self.b_r]))
        # # Equality constraint: Ax = b
        # A =  matrix(np.vstack([self.A_fl]))
        # b =  matrix(np.hstack([self.b_fl]))

        # # Solve QP problem
        # initial_value = self.mmdp.initial_distribution
        # initial_value = occupancy_measures
        # sol_dict = solvers.qp(P, q, G, h, A, b, initvals = initial_value)

        # sol = np.array(sol_dict['x']).flatten()
        # print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        # # sol = result.x
        # # feasibility check
        # if (self.flow_constraint(sol) >= 1e-6).any():
        #     print("***********SOLUTION NOT FEASIBLE*************")
        # deceptive_occupancy_measures = sol.reshape((n_states, n_actions))
        # deceptive_occupancy_measures = np.where((deceptive_occupancy_measures < 0) & (deceptive_occupancy_measures > self.threshold), 0, deceptive_occupancy_measures)
        # denominator = np.repeat(np.sum(deceptive_occupancy_measures, axis = -1).reshape(n_states,1), n_actions, axis = -1)
        # denominator = np.where(denominator >= 1e-9, denominator, 1e-9)
        # deceptive_policy = deceptive_occupancy_measures / denominator
        # deceptive_value_function = policy_evaluation(transitions, r, deceptive_policy, 1e-6, gamma)
        # deceptive_revenue = np.sum(r * deceptive_occupancy_measures)

        # return deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue

        
    
    def targeted_deception(self, target_occupancy_measures, beta = 0):
        """
        Solve MMDP with targeted deception (Optimization Problem 5)
        """

        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        beta = -beta
        gamma = self.mmdp.gamma
        transitions = self.mmdp.transition_matrices
        r = self.mmdp.joint_rewards

        print("Solving Targeted Deception...")
        
        time0 = time.time()

        P = - (2 * beta) * np.eye(n_states * n_actions)
        q = (2 * beta) * target_occupancy_measures.flatten() - r.flatten()

        # Objective function
        P = matrix(P)
        q = matrix(q)
        # Inequality constraint: Gx <= h
        G = matrix(np.vstack([self.A_p, self.A_r]))
        h =  matrix(np.vstack([self.b_p, self.b_r]))
        # Equality constraint: Ax = b
        A =  matrix(np.vstack([self.A_fl]))
        b =  matrix(np.hstack([self.b_fl]))

        # Solve QP problem
        sol_dict = solvers.qp(P, q, G, h, A, b)

        sol = np.array(sol_dict['x']).flatten()
        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        # sol = result.x
        # feasibility check
        if (self.flow_constraint(sol) >= 1e-6).any():
            print("***********SOLUTION NOT FEASIBLE*************")
        deceptive_occupancy_measures = sol.reshape((n_states, n_actions))
        deceptive_occupancy_measures = np.where((deceptive_occupancy_measures < 0) & (deceptive_occupancy_measures > self.threshold), 0, deceptive_occupancy_measures)
        denominator = np.repeat(np.sum(deceptive_occupancy_measures, axis = -1).reshape(n_states,1), n_actions, axis = -1)
        denominator = np.where(denominator >= 1e-9, denominator, 1e-9)
        deceptive_policy = deceptive_occupancy_measures / denominator
        deceptive_value_function = policy_evaluation(transitions, r, deceptive_policy, 1e-6, gamma)
        deceptive_revenue = np.sum(r * deceptive_occupancy_measures)

        return deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue
        
    def equivocal_deception(self):
        """
        Solve MMDP with equivocal deception (Optimization Problem 6)
        """
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        gamma = self.mmdp.gamma
        transitions = self.mmdp.transition_matrices
        r = self.mmdp.joint_rewards


        print("Solving Equivocal Deception...")
        time0 = time.time()
        
        # Objective function
        c = -r.flatten()
        # Inequality constraint: Gx <= h
        G = np.vstack([self.A_p, self.A_r])
        h = np.vstack([self.b_p, self.b_r])
        # Equality constraint: Ax = b
        A = np.vstack([self.A_fl, self.A_eq])
        b = np.hstack([self.b_fl, self.b_eq])

        # Convert the problem into cvxopt format
        G_cvxopt = matrix(G)
        h_cvxopt = matrix(h)
        A_cvxopt = matrix(A)
        b_cvxopt = matrix(b)
        c_cvxopt = matrix(c)
        
        sol_dict = solvers.lp(c_cvxopt, G_cvxopt, h_cvxopt, A_cvxopt, b_cvxopt)
        
        sol = np.array(sol_dict['x']).flatten()
        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        # sol = result.x
        # feasibility check
        if (self.flow_constraint(sol) >= 1e-6).any():
            print("***********SOLUTION NOT FEASIBLE*************")
        if (np.abs(self.reachability_constraint(sol)) >= 1e-6).any():
            print("***********SOLUTION NOT FEASIBLE*************")
        if (self.equivocal_constraint(sol) >= 1e-6).any():
            print("***********SOLUTION NOT FEASIBLE*************")
        deceptive_occupancy_measures = sol.reshape((n_states, n_actions))
        deceptive_occupancy_measures = np.where((deceptive_occupancy_measures < 0) & (deceptive_occupancy_measures > self.threshold), 0, deceptive_occupancy_measures)
        denominator = np.repeat(np.sum(deceptive_occupancy_measures, axis = -1).reshape(n_states,1), n_actions, axis = -1)
        denominator = np.where(denominator >= 1e-9, denominator, 1e-9)
        deceptive_policy = deceptive_occupancy_measures / denominator
        deceptive_value_function = policy_evaluation(transitions, r, deceptive_policy, 1e-6, gamma)
        deceptive_revenue = np.sum(r * deceptive_occupancy_measures)

        return deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue