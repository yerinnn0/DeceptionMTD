import numpy as np

import scipy.sparse as sp
import time
import torch
from scipy.sparse import issparse

import pyomo.environ as pyo

from .policy_optimization import PolicyOptimization


class PolicyOptimizationPyomo(PolicyOptimization):

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

        # violation = torch.sparse.mm(self.A_fl_torch, X_flat.unsqueeze(1)).squeeze() - self.b_fl_torch.reshape((-1,))

        violation = self.A_fl_torch@X_flat - self.b_fl_torch.flatten()
         
        assert len(violation.shape)==1

        # return self.A_fl_torch@X_flat - self.b_fl_torch
        return violation

    def reachability_constraint_matrix(self, X_flat):

        return torch.sparse.mm(self.A_r_torch, X_flat.unsqueeze(1)).squeeze() - self.b_r_torch
        

    def diversionary_deception(self, original_occupancy_measures, beta=1, init=None):
        print("Solving Diversionary Deception (Sparse Optimized)")

        t0 = time.time()

        # Convert to NumPy without keeping GPU + autograd history
        if isinstance(original_occupancy_measures, torch.Tensor):
            original_occupancy_measures = original_occupancy_measures.detach().cpu().numpy()

        n = self.mmdp.n_joint_states * self.mmdp.n_joint_actions
        c = (2 * beta) * original_occupancy_measures.ravel() - self.r.ravel()

        # Assume A_ineq, A_eq are SciPy sparse
        A_ineq = -self.G
        b_ineq = -self.h
        A_eq = self.A
        b_eq = self.b

        if not issparse(A_ineq) or not issparse(A_eq):
            raise ValueError("For memory efficiency, pass A_ineq and A_eq as SciPy sparse matrices.")

        model = pyo.ConcreteModel()
        model.N = pyo.RangeSet(0, n - 1)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)

        # Objective: no dense Q
        def obj_rule(m):
            return -beta * sum(m.x[i] * m.x[i] for i in m.N) + sum(c[i] * m.x[i] for i in m.N)
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Inequality constraints from sparse rows
        m_ineq = A_ineq.shape[0]
        model.M_ineq = pyo.RangeSet(0, m_ineq - 1)
        def ineq_rule(m, k):
            row_start = A_ineq.indptr[k]
            row_end = A_ineq.indptr[k+1]
            cols = A_ineq.indices[row_start:row_end]
            vals = A_ineq.data[row_start:row_end]
            return sum(vals[j] * m.x[cols[j]] for j in range(len(cols))) >= b_ineq[k]
        model.ineq_con = pyo.Constraint(model.M_ineq, rule=ineq_rule)

        # Equality constraints from sparse rows
        m_eq = A_eq.shape[0]
        model.M_eq = pyo.RangeSet(0, m_eq - 1)
        def eq_rule(m, k):
            row_start = A_eq.indptr[k]
            row_end = A_eq.indptr[k+1]
            cols = A_eq.indices[row_start:row_end]
            vals = A_eq.data[row_start:row_end]
            return sum(vals[j] * m.x[cols[j]] for j in range(len(cols))) == b_eq[k]
        model.eq_con = pyo.Constraint(model.M_eq, rule=eq_rule)

        # Solve
        solver = pyo.SolverFactory("ipopt")
        result = solver.solve(model, tee=True)

        print("Time:", time.time() - t0)

        # Extract solution in streaming fashion
        sol = np.fromiter((pyo.value(model.x[i]) for i in model.N), dtype=float, count=n)

        return self.evaluation(sol)


    def targeted_deception(self, target_occupancy_measures, initvals, beta = 0):
        print("Solving Diversionary Deception (Sparse Optimized)")

        t0 = time.time()
        beta = -beta

        # Convert to NumPy without keeping GPU + autograd history
        if isinstance(target_occupancy_measures, torch.Tensor):
            target_occupancy_measures = target_occupancy_measures.detach().cpu().numpy()

        n = self.mmdp.n_joint_states * self.mmdp.n_joint_actions
        c = (2 * beta) * target_occupancy_measures.ravel() - self.r.ravel()

        # Assume A_ineq, A_eq are SciPy sparse
        A_ineq = -self.G
        b_ineq = -self.h
        A_eq = self.A
        b_eq = self.b

        if not issparse(A_ineq) or not issparse(A_eq):
            raise ValueError("For memory efficiency, pass A_ineq and A_eq as SciPy sparse matrices.")

        model = pyo.ConcreteModel()
        model.N = pyo.RangeSet(0, n - 1)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)

        # Objective: no dense Q
        def obj_rule(m):
            return -beta * sum(m.x[i] * m.x[i] for i in m.N) + sum(c[i] * m.x[i] for i in m.N)
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Inequality constraints from sparse rows
        m_ineq = A_ineq.shape[0]
        model.M_ineq = pyo.RangeSet(0, m_ineq - 1)
        def ineq_rule(m, k):
            row_start = A_ineq.indptr[k]
            row_end = A_ineq.indptr[k+1]
            cols = A_ineq.indices[row_start:row_end]
            vals = A_ineq.data[row_start:row_end]
            return sum(vals[j] * m.x[cols[j]] for j in range(len(cols))) >= b_ineq[k]
        model.ineq_con = pyo.Constraint(model.M_ineq, rule=ineq_rule)

        # Equality constraints from sparse rows
        m_eq = A_eq.shape[0]
        model.M_eq = pyo.RangeSet(0, m_eq - 1)
        def eq_rule(m, k):
            row_start = A_eq.indptr[k]
            row_end = A_eq.indptr[k+1]
            cols = A_eq.indices[row_start:row_end]
            vals = A_eq.data[row_start:row_end]
            return sum(vals[j] * m.x[cols[j]] for j in range(len(cols))) == b_eq[k]
        model.eq_con = pyo.Constraint(model.M_eq, rule=eq_rule)

        # Solve
        solver = pyo.SolverFactory("ipopt")
        result = solver.solve(model, tee=True)

        print("Time:", time.time() - t0)

        # Extract solution in streaming fashion
        sol = np.fromiter((pyo.value(model.x[i]) for i in model.N), dtype=float, count=n)

        return self.evaluation(sol)


    def equivocal_deception(self, original_occupancy_measures= None, beta=1, initvals = None):
        print("Solving Diversionary Deception (Sparse Optimized)")

        t0 = time.time()

        # Convert to NumPy without keeping GPU + autograd history
        if isinstance(original_occupancy_measures, torch.Tensor):
            original_occupancy_measures = original_occupancy_measures.detach().cpu().numpy()

        n = self.mmdp.n_joint_states * self.mmdp.n_joint_actions
        
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

        c = -self.r.flatten() - (2*beta*(self.b_eq*self.A_eq)).flatten()

        # Assume A_ineq, A_eq are SciPy sparse
        A_ineq = -self.G
        b_ineq = -self.h
        A_eq = self.A
        b_eq = self.b

        if not issparse(A_ineq) or not issparse(A_eq):
            raise ValueError("For memory efficiency, pass A_ineq and A_eq as SciPy sparse matrices.")

        model = pyo.ConcreteModel()
        model.N = pyo.RangeSet(0, n - 1)
        model.x = pyo.Var(model.N, domain=pyo.NonNegativeReals)

        # Objective: no dense Q
        def obj_rule(m):
            return 0.5 * sum(P[i, j] * m.x[i] * m.x[j] for i in m.N for j in m.N) + \
                sum(c[i] * m.x[i] for i in m.N)
        model.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # Inequality constraints from sparse rows
        m_ineq = A_ineq.shape[0]
        model.M_ineq = pyo.RangeSet(0, m_ineq - 1)
        def ineq_rule(m, k):
            row_start = A_ineq.indptr[k]
            row_end = A_ineq.indptr[k+1]
            cols = A_ineq.indices[row_start:row_end]
            vals = A_ineq.data[row_start:row_end]
            return sum(vals[j] * m.x[cols[j]] for j in range(len(cols))) >= b_ineq[k]
        model.ineq_con = pyo.Constraint(model.M_ineq, rule=ineq_rule)

        # Equality constraints from sparse rows
        m_eq = A_eq.shape[0]
        model.M_eq = pyo.RangeSet(0, m_eq - 1)
        def eq_rule(m, k):
            row_start = A_eq.indptr[k]
            row_end = A_eq.indptr[k+1]
            cols = A_eq.indices[row_start:row_end]
            vals = A_eq.data[row_start:row_end]
            return sum(vals[j] * m.x[cols[j]] for j in range(len(cols))) == b_eq[k]
        model.eq_con = pyo.Constraint(model.M_eq, rule=eq_rule)

        # Solve
        solver = pyo.SolverFactory("ipopt")
        result = solver.solve(model, tee=True)

        print("Time:", time.time() - t0)

        # Extract solution in streaming fashion
        sol = np.fromiter((pyo.value(model.x[i]) for i in model.N), dtype=float, count=n)

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
    
    
    def evaluation(self, occupancy_measure):

        time0 = time.time()
        
        print("Evaluating Solution in ", self.device)

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
        # value_function = self.policy_evaluation(self.transitions_torch, self.r_torch, policy, 1e-5, self.gamma)
        value_function = None

        # Revenue (r must be a tensor on the same device)
        revenue = torch.sum(self.r_torch * occupancy_measures).item()
        
        print("Time for evaluating solution:", time.time()-time0)

        return occupancy_measures, policy, value_function, revenue
        