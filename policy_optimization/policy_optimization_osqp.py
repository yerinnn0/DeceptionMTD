"""
Solve optimization problem for deceptive policy
"""
import numpy as np

import osqp
# from scipy.optimize import minimize
import scipy.sparse as sp
import time
# from cvxopt import matrix, spmatrix, solvers
import torch

from .policy_optimization import PolicyOptimization

# def scipy_to_cvxopt_sparse(sparse_mat):
#     coo = sparse_mat.tocoo()
#     return spmatrix(coo.data, coo.row.tolist(), coo.col.tolist(), size=sparse_mat.shape)


class PolicyOptimizationOSQP(PolicyOptimization):

    def __init__(self, mmdp):
        super().__init__(mmdp)

        
        def scipy_to_torch_sparse(scipy_mat, device='cuda'):
            scipy_coo = scipy_mat.tocoo()
            indices = torch.tensor(np.array([scipy_coo.row, scipy_coo.col]), dtype=torch.long, device=device)
            values = torch.tensor(scipy_coo.data, dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(indices, values, scipy_mat.shape, device=device)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transitions_torch = torch.tensor(self.transitions, dtype=torch.float32, device=self.device)
        self.r_torch = torch.tensor(self.r, dtype=torch.float32, device=self.device)

        self.A_fl_torch = scipy_to_torch_sparse(self.A_fl, self.device)
        self.A_r_torch = scipy_to_torch_sparse(self.A_r, self.device)
        self.A_p_torch = scipy_to_torch_sparse(self.A_p, self.device)
        self.A_eq_torch = scipy_to_torch_sparse(self.A_eq, self.device)

        self.b_fl_torch = torch.tensor(self.b_fl, dtype=torch.float32, device=self.device)
        self.b_r_torch = torch.tensor(self.b_r, dtype=torch.float32, device=self.device)
        self.b_p_torch = torch.tensor(self.b_p, dtype=torch.float32, device=self.device)
        self.b_eq_torch = torch.tensor(self.b_eq, dtype=torch.float32, device=self.device)

        
    def flow_constraint_matrix(self, X_flat):

        violation = torch.sparse.mm(self.A_fl_torch, X_flat.unsqueeze(1)).squeeze() - self.b_fl_torch.reshape((-1,))

        assert len(violation.shape)==1

        # return self.A_fl_torch@X_flat - self.b_fl_torch
        return violation

    def reachability_constraint_matrix(self, X_flat):

        return torch.sparse.mm(self.A_r_torch, X_flat.unsqueeze(1)).squeeze() - self.b_r_torch

        
    def solve_MDP(self):
        """
        Solve MMDP without deception (Optimization Problem 3)
        """

        print("Solving MDP...")
        
        time0 = time.time()

        n = self.n_states * self.n_actions  # Size of the matrix
        P = P = sp.bsr_matrix((n, n))
        q = - self.r.flatten()
        
        # Use slack variables for equality constraint
        A = sp.vstack([self.A_p, self.A_r,self.A_fl], format = "csc")
        l = np.vstack([-np.inf*np.ones(self.b_p.shape), -np.inf*np.ones(self.b_r.shape),self.b_fl])
        u = np.vstack([self.b_p, self.b_r,self.b_fl])
        
        # Solve QP problem: OSQP
        solver = osqp.OSQP(algebra='cuda')
        solver.setup(P, q, A, l, u, eps_abs = 1e-5, eps_rel = 1e-5, 
                     eps_dual_inf = 1e-5, eps_prim_inf = 1e-5, alpha = 1, 
                     polish = True, verbose = False)
        result = solver.solve()
        sol = (result.x).flatten()
        
        print("Time :", time.time()-time0)

        return self.evaluation(sol)
        
    def targeted_deception(self, target_occupancy_measures, initvals, beta = 0):
        """
        Solve MMDP with targeted deception (Optimization Problem 5)
        """

        beta = -beta

        print("Solving Targeted Deception...")
        
        time0 = time.time()

        # Objective function
        n = self.n_states * self.n_actions  # Size of the matrix
        # diagonal_values = [-2 * beta] * n  # Diagonal entries (-2 * beta)
        # P = sp.diags(diagonal_values, format="csc")
        # q = np.asarray((2 * beta) * target_occupancy_measures.flatten() - self.r.flatten())

        # # Inequality constraint: l <= Ax <= u
        # A = sp.vstack([self.A_p, self.A_r,self.A_fl], format = "csc")
        # l = np.vstack([-np.inf*np.ones(self.b_p.shape), -np.inf*np.ones(self.b_r.shape),self.b_fl])
        # u = np.vstack([self.b_p, self.b_r,self.b_fl])

        diagonal_values = np.full(n, -2 * beta, dtype=np.float64)  # float64 diagonal values
        P = sp.diags(diagonal_values, format="csc").astype(np.float64)

        q = ((2 * beta) * target_occupancy_measures.flatten() - self.r.flatten()).astype(np.float64)

        A = sp.vstack([
            self.A_p.astype(np.float64),
            self.A_r.astype(np.float64),
            self.A_fl.astype(np.float64)
        ], format="csc")

        l = np.vstack([
            -np.inf * np.ones(self.b_p.shape, dtype=np.float64),
            -np.inf * np.ones(self.b_r.shape, dtype=np.float64),
            self.b_fl.astype(np.float64)
        ])

        u = np.vstack([
            self.b_p.astype(np.float64),
            self.b_r.astype(np.float64),
            self.b_fl.astype(np.float64)
        ])
        
        # Solve QP problem: OSQP
        solver = osqp.OSQP(algebra='cuda')
        # solver.setup(P, q, A, l, u, eps_abs = 1e-8, eps_rel = 1e-8, 
        #              eps_dual_inf = 1e-8, eps_prim_inf = 1e-8, alpha = 1, 
        #              polish = True, verbose = False, warm_start=True)
        solver.setup(P, q, A, l, u,
            eps_abs=1e-5, eps_rel=1e-5,
            eps_dual_inf=1e-5, eps_prim_inf=1e-5,
            polish=True, alpha=1.6,
            rho=0.01, adaptive_rho=False, verbose=False, warm_start=True)
        result = solver.solve()
        sol = (result.x).flatten()

        print(result.info.status)           # 'solved', 'primal infeasible', etc.

        print("Primal residual:", result.info.prim_res)
        print("Dual residual:", result.info.dual_res)
        
        print("Time :", time.time()-time0)

        return self.evaluation(sol)
    
    # def equivocal_deception(self):
    #     """
    #     Solve MMDP with equivocal deception (Optimization Problem 6)
    #     """

    #     print("Solving Equivocal Deception...")
        
    #     time0 = time.time()

    #     # Objective function
    #     n = self.n_states * self.n_actions  # Size of the matrix
    #     P = np.zeros((n,n))
    #     P = sp.bsr_matrix(P)
    #     q = - self.r.flatten()

    #     # Inequality constraint: l <= Ax <= u
    #     A = sp.vstack([self.A_p, self.A_r,self.A_fl, self.A_eq], format = "csc")
    #     l = np.vstack([-np.inf*np.ones(self.b_p.shape), -np.inf*np.ones(self.b_r.shape),self.b_fl, self.b_eq])
    #     u = np.vstack([self.b_p, self.b_r,self.b_fl,self.b_eq])
        
    #     # Solve QP problem: OSQP
    #     solver = osqp.OSQP()
    #     solver.setup(P, q, A, l, u, eps_abs = 1e-8, eps_rel = 1e-8, 
    #                  eps_dual_inf = 1e-8, eps_prim_inf = 1e-8, alpha = 1, 
    #                  polish = True, verbose = False)
    #     result = solver.solve()
    #     sol = (result.x).flatten()
        
    #     print("Time :", time.time()-time0)

    #     return self.evaluation(sol)
        
        
    def equivocal_deception(self, beta = 1, initvals = None):
        """
        Solve MMDP with equivocal deception (Optimization Problem 6 Modified)
        """

        time0 = time.time()

        # Objective function
        if self.n_states < 1000:
             P = 2*beta*self.A_eq.T.dot(self.A_eq)
             P = sp.bsr_matrix(P) 
        else:
            v = self.A_eq.T.tocsc()
            # v.eliminate_zeros()
            # P = 2 * beta * v.dot(v.T)
            # P.data = P.data.astype(np.float32)

            # Create an empty diagonal
            diag_P = np.zeros(v.shape[0], dtype=np.float32)

            # Square each nonzero and assign to diagonal
            diag_P[v.indices] = 2 * beta * np.power(v.data, 2)

            # Build sparse diagonal matrix
            P = sp.diags(diag_P, format='csc')

            # print(f"After thresholding, v nnz: {v.nnz} / {v.shape[0]}")
            # print(f"v shape: {v.shape}, nnz: {v.nnz}")
            # print(f"Sparsity of v: {100 * v.nnz / v.shape[0]:.5f}%")
            # print(f"P shape: {P.shape}, nnz: {P.nnz}, approx size (MB): {P.data.nbytes / 1e6:.2f}")

        q = -self.r.flatten() - (2*beta*(self.b_eq*self.A_eq)).flatten()

        # Inequality constraint: l <= Ax <= u
        A = sp.vstack([self.A_p, self.A_r,self.A_fl], format = "csc")
        l = np.vstack([-np.inf*np.ones(self.b_p.shape), -np.inf*np.ones(self.b_r.shape),self.b_fl]).ravel()
        u = np.vstack([self.b_p, self.b_r,self.b_fl]).ravel()

        if isinstance(initvals,torch.Tensor):
            initvals =  initvals.cpu().numpy()
        print(isinstance(initvals, np.ndarray), isinstance(initvals, torch.Tensor))
        
        print("Solving Equivocal Deception with OSQP...")
        
        # Solve QP problem: OSQP
        solver = osqp.OSQP(algebra='cuda')
        solver.setup(P, q, A, l, u,
            eps_abs=1e-5, eps_rel=1e-5,
            eps_dual_inf=1e-5, eps_prim_inf=1e-5,
            polish=True, alpha=1.6,
            rho=0.01, adaptive_rho=False, verbose=False, warm_start=True)
        # solver.warm_start(initvals)
        result = solver.solve()
        sol = (result.x).flatten()

        print(result.info.status)           # 'solved', 'primal infeasible', etc.

        print("Primal residual:", result.info.prim_res)
        print("Dual residual:", result.info.dual_res)

        print("Time for solving equivocal deception :", time.time()-time0)

        return self.evaluation(sol)
    
    def policy_evaluation(self, transitions, rewards, policy, eta, gamma, chunk_size=1000):
        V = torch.zeros(self.n_states, device=self.device)
        delta = torch.tensor(float('inf'), device=self.device)

        S, _, _ = transitions.shape

        while delta > eta:
            V_next_chunks = []

            for start in range(0, S, chunk_size):
                end = min(start + chunk_size, S)
                # chunk 크기만큼만 계산
                # transitions_chunk: (chunk_size, A, S)
                transitions_chunk = transitions[start:end, :, :]
                # V.view(1,1,S) broadcasting 필요 -> (1,1,S)
                # 곱하고 sum(dim=2) -> (chunk_size, A)
                chunk_result = (transitions_chunk * V.view(1, 1, self.n_states)).sum(dim=2)
                V_next_chunks.append(chunk_result)

            # (S, A) 크기로 다시 합침
            V_next = torch.cat(V_next_chunks, dim=0)

            Q_sa = rewards + gamma * V_next  # (S, A)
            V_new = (policy * Q_sa).sum(dim=1)  # (S,)

            delta = torch.max(torch.abs(V - V_new))
            V = V_new

        return V

    

    # def policy_evaluation(self, transitions, rewards, policy, eta, gamma):

    #     V = torch.zeros(self.n_states, device=self.device)
    #     delta = torch.tensor(float('inf'), device=self.device)

    #     while delta > eta:
    #         # V[s'] broadcast to (S, A, S)
    #         V_next = (transitions * V.view(1, 1, self.n_states)).sum(dim=2)  # shape: (S, A)
    #         Q_sa = rewards + gamma * V_next  # (S, A)
    #         V_new = (policy * Q_sa).sum(dim=1)  # (S,)
            
    #         delta = torch.max(torch.abs(V - V_new))
    #         V = V_new

    #     return V  # shape: (S,)

    
    def evaluation(self, occupancy_measure):

        time0 = time.time()

        # Convert to tensor on GPU
        occupancy_measure = torch.tensor(occupancy_measure, dtype=torch.float32, device=self.device)

        # Check and fix occupancy_measure
        if torch.any(occupancy_measure < -1e-2):
            print("***********NON-NEGATIVITY VIOLATED*************")
            print(torch.min(occupancy_measure).item())

        occupancy_measure = torch.clamp(occupancy_measure, min=0.0)

        # Flow constraint (assumes it returns a torch tensor on the same device)
        flow_violation = self.flow_constraint_matrix(occupancy_measure)
        reach_violation = self.reachability_constraint_matrix(occupancy_measure)

        if torch.max(torch.abs(flow_violation)) >= 1e-4:
            print("***********FLOW CONSTRAINT VIOLATED*************")
            print(torch.max(torch.abs(flow_violation)).item())

        if torch.max(reach_violation) > 1e-4:
            print("***********TASK CONSTRAINT VIOLATED*************")
            print(torch.max(reach_violation).item())

        # Normalize to policy
        occupancy_measures = occupancy_measure.view(self.n_states, self.n_actions)
        occupancy_measures = torch.clamp(occupancy_measures, min=1e-9)

        row_sums = torch.sum(occupancy_measures, dim=1, keepdim=True)
        row_sums = torch.clamp(row_sums, min=1e-9)

        policy = occupancy_measures / row_sums

        # Compute value function (assumes self.policy_evaluation supports torch)
        value_function = self.policy_evaluation(self.transitions_torch, self.r_torch, policy, 1e-6, self.gamma)

        # Revenue (r must be a tensor on the same device)
        revenue = torch.sum(self.r_torch * occupancy_measures).item()
        
        print("Time for evaluating solution:", time.time()-time0)

        return occupancy_measures, policy, value_function, revenue
        