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


class PolicyOptimizationGD(PolicyOptimization):

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
    
    def diversionary_deception(self, target_occupancy_measures, init, beta=0, lr=1e-3, max_iters=1000, tol=1e-5):
        """
        Solve MMDP with targeted deception using Projected Gradient Descent (PGD).
        Uses float64 and GPU acceleration.
        Projects x back into l <= A x <= u space after each step.
        """

        # beta = -beta
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float64

        print(f"Solving Targeted Deception with Projected Gradient Descent (float64 on {device})...")

        n = self.n_states * self.n_actions
        diagonal_values = torch.full((n,), -2 * beta, dtype=dtype, device=device)
        P = torch.diag(diagonal_values)

        q = (2 * beta) * torch.tensor(target_occupancy_measures.flatten(), dtype=dtype, device=device) \
            - torch.tensor(self.r.flatten(), dtype=dtype, device=device)

        A_p = torch.tensor(self.A_p.toarray(), dtype=dtype, device=device)
        A_r = torch.tensor(self.A_r.toarray(), dtype=dtype, device=device)
        A_fl = torch.tensor(self.A_fl.toarray(), dtype=dtype, device=device)
        A = torch.cat([A_p, A_r, A_fl], dim=0)

        l = torch.cat([
            -torch.full((self.b_p.shape[0],), float('inf'), dtype=dtype, device=device),
            -torch.full((self.b_r.shape[0],), float('inf'), dtype=dtype, device=device),
            torch.tensor(self.b_fl.flatten(), dtype=dtype, device=device)
        ], dim=0)

        u = torch.cat([
            torch.tensor(self.b_p.flatten(), dtype=dtype, device=device),
            torch.tensor(self.b_r.flatten(), dtype=dtype, device=device),
            torch.tensor(self.b_fl.flatten(), dtype=dtype, device=device)
        ], dim=0)

        x = torch.tensor(init.flatten(), dtype=dtype, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=lr)

        def project_onto_linear_constraints(x, A, l, u):
            """
            Project x onto the set {x : l <= Ax <= u} by solving:
            min ||x' - x||^2 s.t. l <= Ax' <= u
            """
            with torch.no_grad():
                # Use least-squares solve with clamped Ax
                Ax = A @ x
                Ax_clamped = torch.clamp(Ax, l, u)

                # Solve least-squares problem: min ||A x' - Ax_clamped||^2
                # Solution: x' = (A^T A)^-1 A^T Ax_clamped
                AtA = A.T @ A
                Atb = A.T @ Ax_clamped
                x_proj = torch.linalg.solve(AtA + 1e-10 * torch.eye(AtA.shape[0], device=device, dtype=dtype), Atb)

                return x_proj

        for iter in range(max_iters):
            optimizer.zero_grad()

            obj = 0.5 * torch.matmul(x, torch.matmul(P, x)) + torch.dot(q, x)
            loss = obj
            loss.backward()

            grad_norm = x.grad.norm().item()
            optimizer.step()

            # Project back to feasible region
            x.data = project_onto_linear_constraints(x.data, A, l, u)

            # Compute feasibility residual
            with torch.no_grad():
                Ax = A @ x
                constraint_violation = torch.cat([
                    torch.clamp(l - Ax, min=0),
                    torch.clamp(Ax - u, min=0)
                ])
                prim_res = constraint_violation.max().item()

            if iter % 100 == 0 or iter == max_iters - 1:
                print(f"Iter {iter:4d}: obj={obj.item():.6e}, grad_norm={grad_norm:.2e}, prim_res={prim_res:.2e}")

            if grad_norm < tol and prim_res < tol:
                print("Converged!")
                break

        sol = x.detach().cpu().numpy().flatten()

        sol = self.project_onto_feasible_set(sol, A, l, u)

        return self.evaluation(sol)
        

    def targeted_deception(self, target_occupancy_measures, initvals, beta=0, lr=1e-3, max_iters=1000, tol=1e-5):
        """
        Solve MMDP with targeted deception using Projected Gradient Descent (PGD).
        Uses float64 and GPU acceleration.
        Projects x back into l <= A x <= u space after each step.
        """

        beta = -beta
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dtype = torch.float64

        print(f"Solving Targeted Deception with Projected Gradient Descent (float64 on {device})...")

        n = self.n_states * self.n_actions
        diagonal_values = torch.full((n,), -2 * beta, dtype=dtype, device=device)
        P = torch.diag(diagonal_values)

        q = (2 * beta) * torch.tensor(target_occupancy_measures.flatten(), dtype=dtype, device=device) \
            - torch.tensor(self.r.flatten(), dtype=dtype, device=device)

        A_p = torch.tensor(self.A_p.toarray(), dtype=dtype, device=device)
        A_r = torch.tensor(self.A_r.toarray(), dtype=dtype, device=device)
        A_fl = torch.tensor(self.A_fl.toarray(), dtype=dtype, device=device)
        A = torch.cat([A_p, A_r, A_fl], dim=0)

        l = torch.cat([
            -torch.full((self.b_p.shape[0],), float('inf'), dtype=dtype, device=device),
            -torch.full((self.b_r.shape[0],), float('inf'), dtype=dtype, device=device),
            torch.tensor(self.b_fl.flatten(), dtype=dtype, device=device)
        ], dim=0)

        u = torch.cat([
            torch.tensor(self.b_p.flatten(), dtype=dtype, device=device),
            torch.tensor(self.b_r.flatten(), dtype=dtype, device=device),
            torch.tensor(self.b_fl.flatten(), dtype=dtype, device=device)
        ], dim=0)

        x = torch.tensor(initvals.flatten(), dtype=dtype, device=device, requires_grad=True)
        optimizer = torch.optim.Adam([x], lr=lr)

        def project_onto_linear_constraints(x, A, l, u):
            """
            Project x onto the set {x : l <= Ax <= u} by solving:
            min ||x' - x||^2 s.t. l <= Ax' <= u
            """
            with torch.no_grad():
                # Use least-squares solve with clamped Ax
                Ax = A @ x
                Ax_clamped = torch.clamp(Ax, l, u)

                # Solve least-squares problem: min ||A x' - Ax_clamped||^2
                # Solution: x' = (A^T A)^-1 A^T Ax_clamped
                AtA = A.T @ A
                Atb = A.T @ Ax_clamped
                x_proj = torch.linalg.solve(AtA + 1e-10 * torch.eye(AtA.shape[0], device=device, dtype=dtype), Atb)

                return x_proj

        for iter in range(max_iters):
            optimizer.zero_grad()

            obj = 0.5 * torch.matmul(x, torch.matmul(P, x)) + torch.dot(q, x)
            loss = obj
            loss.backward()

            grad_norm = x.grad.norm().item()
            optimizer.step()

            # Project back to feasible region
            x.data = project_onto_linear_constraints(x.data, A, l, u)

            # Compute feasibility residual
            with torch.no_grad():
                Ax = A @ x
                constraint_violation = torch.cat([
                    torch.clamp(l - Ax, min=0),
                    torch.clamp(Ax - u, min=0)
                ])
                prim_res = constraint_violation.max().item()

            if iter % 100 == 0 or iter == max_iters - 1:
                print(f"Iter {iter:4d}: obj={obj.item():.6e}, grad_norm={grad_norm:.2e}, prim_res={prim_res:.2e}")

            if grad_norm < tol and prim_res < tol:
                print("Converged!")
                break

        sol = x.detach().cpu().numpy().flatten()

        sol = self.project_onto_feasible_set(sol, A, l, u)

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
            # P = 2 * gamma * v.dot(v.T)
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
        
        print("Solving Equivocal Deception...")
        
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
    
    def project_onto_feasible_set(x_input, A, l, u):
        """
        Project x_input onto the feasible set {x | l <= A x <= u}
        by solving a QP:
            min_x 0.5 * ||x - x_input||^2
            s.t.   l <= A x <= u

        Inputs:
            x_input : (n,) torch tensor, input point (float64, on GPU)
            A       : (m, n) torch tensor
            l, u    : (m,) torch tensors

        Output:
            x_proj : (n,) projected point
        """

        # Move tensors to CPU and convert to numpy (OSQP works on CPU)
        x_np = x_input.detach().cpu().numpy()
        A_np = A.detach().cpu().numpy()
        l_np = l.detach().cpu().numpy()
        u_np = u.detach().cpu().numpy()

        n = x_np.shape[0]

        # QP Form:
        # min 0.5 * x^T I x - x_input^T x
        # s.t. l <= A x <= u

        P = sp.eye(n, format='csc', dtype=np.float64)
        q = -x_np.astype(np.float64)

        A_sparse = sp.csc_matrix(A_np.astype(np.float64))

        # Setup OSQP
        solver = osqp.OSQP()
        solver.setup(P=P, q=q, A=A_sparse, l=l_np, u=u_np, verbose=False, eps_abs=1e-10, eps_rel=1e-10)
        result = solver.solve()

        if result.info.status != 'solved':
            raise RuntimeError(f"Projection QP failed: {result.info.status}")

        x_proj_np = result.x.astype(np.float64)

        # Convert back to torch on original device
        x_proj = torch.tensor(x_proj_np, dtype=torch.float64, device=x_input.device)

        return x_proj

        