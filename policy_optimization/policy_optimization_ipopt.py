"""
Solve optimization problem for deceptive policy
"""
import numpy as np
import scipy.sparse as sp
from cvxopt import matrix, spmatrix, solvers
from cyipopt import Problem

import time


from .policy_optimization import PolicyOptimization


def scipy_to_cvxopt_sparse(sparse_mat):
    coo = sparse_mat.tocoo()
    return spmatrix(coo.data, coo.row.tolist(), coo.col.tolist(), size=sparse_mat.shape)

class PolicyOptimizationIPOPT(PolicyOptimization):

    def __init__(self, mmdp):
        super().__init__(mmdp)

        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        gamma = self.mmdp.gamma
        transitions = self.mmdp.transition_matrices
        
    def solve_MDP(self):
        """
        Solve MMDP without deception (Optimization Problem 3)
        """
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        r = self.mmdp.joint_rewards
        
        class MDP_Problem(object):
            def __init__(self, A_fl, b_fl, A_r, b_r):
                # Number of variables and constraints
                self.num_variables = A_fl.shape[1]
                self.num_constraints = A_fl.shape[0]+A_r.shape[0]
                #super().__init__(self.num_variables, self.num_constraints)
                
                self.A_fl = A_fl
                self.b_fl = b_fl
                self.A_r = A_r
                self.b_r = b_r
                
            def objective(self, x):
                
                return -(r.flatten().T).dot(x)
            
            def gradient(self, x):
                
                return -r.flatten()
            
            def constraints(self, x):
                
                return np.concatenate([
                    ((self.A_fl).dot(x).flatten()).reshape([-1,1]),
                    ((self.A_r).dot(x).flatten()).reshape([-1,1])
                ], axis = 0).flatten()
            
            def jacobian(self, x):
                
                return sp.vstack((self.A_fl, self.A_r)).toarray().flatten()
            
            # def hessianstructure(self):
            #     """Returns the row and column indices for non-zero vales of the
            #     Hessian."""

            #     # NOTE: The default hessian structure is of a lower triangular matrix,
            #     # therefore this function is redundant. It is included as an example
            #     # for structure callback.

            #     return np.nonzero(np.tril(np.ones((self.num_variables, self.num_variables))))
            
            # def hessian(self,x):
                
            #     H = np.zeros((self.num_variables, self.num_variables))
                
            #     row, col = self.hessianstructure()
                
            #     return H[row, col]
            
        print("Solving MDP...")
        time0 = time.time()
        
        num_variables = n_states * n_actions
        num_constraints = self.A_fl.shape[0]+self.A_r.shape[0]
        # Variable bounds
        lb = np.zeros((num_variables))
        ub = np.inf*np.ones((num_variables))
        # Constraint bounds (flow constraint, reachability constraint)
        cl = np.vstack([self.b_fl, -np.inf*np.ones((self.b_r.shape))]).flatten()
        cu = np.vstack([self.b_fl, self.b_r]).flatten()
        
        # print(cl.shape, cu.shape, "Cl")
        # print(lb.shape, ub.shape)
            
        initial_guess = np.ones((num_variables)).flatten()
        problem_obj = MDP_Problem(self.A_fl, self.b_fl, self.A_r, self.b_r)
        
        mdp_problem = Problem(
            n = num_variables, 
            m = num_constraints,
            problem_obj = problem_obj,
            lb = lb,
            ub = ub,
            cl = cl,
            cu = cu
        )
                
        
        mdp_problem.add_option('sb', 'yes')
        mdp_problem.add_option('print_level', 0)
        mdp_problem.add_option('constr_viol_tol', 1e-8)
        mdp_problem.add_option('acceptable_constr_viol_tol', 1e-8)
        mdp_problem.add_option('hessian_approximation', 'exact')
        #mdp_problem.add_option('mu_strategy', 'adaptive')
        mdp_problem.add_option('derivative_test', 'first-order')
                
        sol, info = mdp_problem.solve(initial_guess)
        
        print(info)
        
        print("Objective shape:", problem_obj.objective(sol).shape)
        print("Gradient shape: ", problem_obj.gradient(sol).shape)
        print("Constraints shape: ", problem_obj.constraints(sol).shape)
        print("Jacobian shape: ", problem_obj.jacobian(sol).shape)
        # print("Hessian shape: ", problem_obj.hessian(sol).shape)
        
        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        return self.evaluation(sol)
    
    
    def diversionary_deception(self, occupancy_measures, beta = 1):
        """
        Solve MMDP with diversionary deception (Optimization Problem 4)
        """
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        beta = beta
        r = self.mmdp.joint_rewards
        
        class Diversionary_Problem(object):
            def __init__(self, A_fl, b_fl, A_r, b_r):
                # Number of variables and constraints
                self.num_variables = A_fl.shape[1]
                self.num_constraints = A_fl.shape[0]+A_r.shape[0]
                #super().__init__(self.num_variables, self.num_constraints)
                
                n = n_states * n_actions  # Size of the matrix
                self.P = sp.diags([-2*beta],[0], shape = (n,n), format = "csr")
                self.q = (2 * beta) * occupancy_measures.flatten() - r.flatten()
                
                self.A_fl = A_fl
                self.b_fl = b_fl
                self.A_r = A_r
                self.b_r = b_r
                
            def objective(self, x):
                
                return (1/2)*x.T.dot(self.P.dot(x)) - self.q.dot(x)
                
            def gradient(self, x):
                
                return (1/2)*(self.P + self.P.T).dot(x) +self.q
            
            def constraints(self, x):
                
                return np.concatenate([
                    ((self.A_fl).dot(x).flatten()).reshape([-1,1]),
                    ((self.A_r).dot(x).flatten()).reshape([-1,1])
                ], axis = 0).flatten()
            
            def jacobian(self, x):
                
                return sp.vstack((self.A_fl, self.A_r)).toarray().flatten()
            
            def hessian(self,x):
                
                return (1/2)*(self.P + self.P.T).flatten()
            
        print("Solving MDP...")
        time0 = time.time()
        
        num_variables = n_states * n_actions
        num_constraints = self.A_fl.shape[0]+self.A_r.shape[0]
        # Variable bounds
        lb = np.zeros((num_variables))
        ub = np.inf*np.ones((num_variables))
        # Constraint bounds (flow constraint, reachability constraint)
        cl = np.vstack([self.b_fl, -np.inf*np.ones((self.b_r.shape))])
        cu = np.vstack([self.b_fl, self.b_r])
            
        initial_guess = np.ones((num_variables)).flatten()
        problem_obj = Diversionary_Problem(self.A_fl, self.b_fl, self.A_r, self.b_r)
        
        mdp_problem = Problem(
            n = num_variables, 
            m = num_constraints,
            problem_obj = problem_obj,
            lb = lb,
            ub = ub,
            cl = cl,
            cu = cu
        )
        
        mdp_problem.add_option('sb', 'yes')
        mdp_problem.add_option('print_level', 0)
        mdp_problem.add_option('tol', 1e-8)
        mdp_problem.add_option('constr_viol_tol', 1e-8)
        mdp_problem.add_option('acceptable_tol', 1e-8)
        mdp_problem.add_option('acceptable_constr_viol_tol', 1e-8)
        mdp_problem.add_option('hessian_approximation', 'exact')
        #mdp_problem.add_option('mu_strategy', 'adaptive')
        mdp_problem.add_option('derivative_test', 'first-order')
                
        sol, info = mdp_problem.solve(initial_guess)
        
        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        return self.evaluation(sol)
    
    def targeted_deception(self, target_occupancy_measures, initvals, beta = 0):
        """
        Solve MMDP with targeted deception (Optimization Problem 5)
        """

        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        beta = -beta
        r = self.mmdp.joint_rewards

        print("Solving Targeted Deception...")
        
        time0 = time.time()
        
        def targeted_objective(X_flat):
            X = X_flat.reshape((n_states, n_actions))

            return -np.sum(beta * X**2 + (r - 2 * beta * target_occupancy_measures) * X)
        
        initial_guess = np.ones(n_states*n_actions)
        
        constraints = [{'type': 'eq', 'fun': self.flow_constraint}, {'type': 'ineq', 'fun': self.reachability_constraint}]
        
        result = minimize(targeted_objective, x0 = initial_guess, constraints = constraints, method = 'ipopt')
        sol = result.x
        
        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        return self.evaluation(sol)
        
    def equivocal_deception(self):
        """
        Solve MMDP with equivocal deception (Optimization Problem 6)
        """
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        r = self.mmdp.joint_rewards

        print("Solving Equivocal Deception...")
        time0 = time.time()
        
        def equivocal_objective(X_flat):
            X = X_flat.reshape((n_states, n_actions))
            return -np.sum(r * X)
        
        initial_guess = np.ones(n_states*n_actions)
        
        constraints = [{'type': 'eq', 'fun': self.flow_constraint}, 
                       {'type': 'ineq', 'fun': self.reachability_constraint}, 
                       {'type': 'eq', 'fun': self.equivocal_constraint}]

        result = minimize(equivocal_objective, x0 = initial_guess, constraints = constraints, method = 'ipopt')
        sol = result.x

        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        return self.evaluation(sol)