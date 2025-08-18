"""
Solve optimization problem for deceptive policy
"""
import numpy as np
import scipy.sparse as sp
import time
from cvxopt import matrix, spmatrix, solvers

from .policy_optimization import PolicyOptimization

def scipy_to_cvxopt_sparse(sparse_mat):
    coo = sparse_mat.tocoo()
    return spmatrix(coo.data, coo.row.tolist(), coo.col.tolist(), size=sparse_mat.shape)


class PolicyOptimizationCVXOPT(PolicyOptimization):

    def __init__(self, mmdp):
        super().__init__(mmdp)

        self.threshold = 1e-3
        solvers.options['show_progress'] = False

    
    def solve_MDP(self):
        """
        Solve MMDP without deception (Optimization Problem 3)
        """
        n_states = self.mmdp.n_joint_states
        r = self.mmdp.joint_rewards

        print("Solving LP...")
        time0 = time.time()
        
        c = -r.flatten()
        # Inequality constraint: Gx <= h
        G = sp.vstack([self.A_p, self.A_r])
        h = np.vstack([self.b_p, self.b_r]).flatten()
        # Equality constraint: Ax = b
        A = self.A_fl
        b = self.b_fl.flatten()

        # Convert the problem into cvxopt format
        G_cvxopt = scipy_to_cvxopt_sparse(G)
        A_cvxopt = scipy_to_cvxopt_sparse(A)
        h_cvxopt = matrix(h.astype(np.float64), tc='d')
        b_cvxopt = matrix(b.astype(np.float64), tc='d')
        c_cvxopt = matrix(c.astype(np.float64), tc='d')
        
        solvers.options['LPsolver'] = 'glpk' # faster for larger problems
        sol_dict = solvers.lp(c_cvxopt, G_cvxopt, h_cvxopt, A_cvxopt, b_cvxopt)
        
        sol = np.array(sol_dict['x']).flatten()
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

        # P = - (2 * beta) * np.eye(n_states * n_actions)
        n = n_states * n_actions  # Size of the matrix
        diagonal_values = [-2 * beta] * n  # Diagonal entries (-2 * beta)
        P  = spmatrix(diagonal_values, range(n), range(n), size=(n, n))
        q = (2 * beta) * target_occupancy_measures.flatten() - r.flatten()

        # Objective function
        # P = matrix(P)
        q = matrix(q)
        # Inequality constraint: Gx <= h
        G = scipy_to_cvxopt_sparse(sp.vstack([self.A_p, self.A_r]))
        h =  matrix(np.vstack([self.b_p, self.b_r]).flatten(), tc='d')
        # Equality constraint: Ax = b
        A =  scipy_to_cvxopt_sparse(self.A_fl)
        b =  matrix(self.b_fl.flatten(), tc='d')

        # Solve QP problem
        solvers.options['warm_start'] = True
        sol_dict = solvers.qp(P, q, G, h, A, b, initvals = matrix(initvals))

        sol = np.array(sol_dict['x']).flatten()
        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        return self.evaluation(sol)
        
    # def equivocal_deception(self):
    #     """
    #     Solve MMDP with equivocal deception (Optimization Problem 6)
    #     """
    #     n_states = self.mmdp.n_joint_states
    #     r = self.mmdp.joint_rewards

    #     print("Solving Equivocal Deception...")
    #     time0 = time.time()
        
    #     # Objective function
    #     c = -r.flatten()
    #     # Inequality constraint: Gx <= h
    #     G = scipy_to_cvxopt_sparse(sp.vstack([self.A_p, self.A_r]))
    #     h = np.vstack([self.b_p, self.b_r])
    #     # Equality constraint: Ax = b
    #     A = scipy_to_cvxopt_sparse(sp.vstack([self.A_fl, self.A_eq]))
    #     b = np.vstack([self.b_fl, self.b_eq])

    #     # Convert the problem into cvxopt format
    #     G_cvxopt = matrix(G)
    #     h_cvxopt = matrix(h.flatten(), tc='d')
    #     A_cvxopt = matrix(A)
    #     b_cvxopt = matrix(b.flatten(), tc='d')
    #     c_cvxopt = matrix(c.flatten(), tc='d')
        
    #     solvers.options['LPsolver'] = 'glpk' # faster for larger problems
    #     sol_dict = solvers.lp(c_cvxopt, G_cvxopt, h_cvxopt, A_cvxopt, b_cvxopt)
        
    #     sol = np.array(sol_dict['x']).flatten()
    #     print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

    #     return self.evaluation(sol)
    
    def equivocal_deception(self, initvals = None, beta = 1):
        """
        Solve MMDP with equivocal deception (Optimization Problem 6 Modified)
        """
        n_states = self.mmdp.n_joint_states
        r = self.mmdp.joint_rewards

        print("Solving Equivocal Deception with CVXOPT...")
        time0 = time.time()
        
        # Objective function
        P = 2*beta*self.A_eq.T.dot(self.A_eq)
        q = -r.flatten() - (2*beta*(self.b_eq*self.A_eq)).flatten()
        P = scipy_to_cvxopt_sparse(P)
        q = matrix(q)
        # Inequality constraint: Gx <= h
        G = scipy_to_cvxopt_sparse(sp.vstack([self.A_p, self.A_r]))
        h =  matrix(np.vstack([self.b_p, self.b_r]).flatten(), tc='d')
        # Equality constraint: Ax = b
        A =  scipy_to_cvxopt_sparse(self.A_fl)
        b =  matrix(self.b_fl.flatten(), tc='d')

        # Solve QP problem
        solvers.options['warm_start'] = True
        sol_dict = solvers.qp(P, q, G, h, A, b, initvals = matrix(initvals))

        sol = np.array(sol_dict['x']).flatten()
        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        return self.evaluation(sol)