import numpy as np

import scipy.sparse as sp
import time
import torch
from qpth.qp import QPFunction


from .policy_optimization import PolicyOptimization


class PolicyOptimizationQPTH(PolicyOptimization):

    def __init__(self, mmdp):
        super().__init__(mmdp)

        
        def scipy_to_torch_sparse(scipy_mat, device='cuda'):
            scipy_coo = scipy_mat.tocoo()
            indices = torch.tensor(np.array([scipy_coo.row, scipy_coo.col]), dtype=torch.long, device=device)
            values = torch.tensor(scipy_coo.data, dtype=torch.float64, device=device)
            return torch.sparse_coo_tensor(indices, values, scipy_mat.shape, device=device)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.transitions_torch = torch.tensor(self.transitions, dtype=torch.float64, device=self.device)
        self.r_torch = torch.tensor(self.r, dtype=torch.float64, device=self.device)

        self.A_fl_torch = scipy_to_torch_sparse(self.A_fl, self.device)
        self.A_r_torch = scipy_to_torch_sparse(self.A_r, self.device)
        self.A_p_torch = scipy_to_torch_sparse(self.A_p, self.device)
        self.A_eq_torch = scipy_to_torch_sparse(self.A_eq, self.device)

        self.b_fl_torch = torch.tensor(self.b_fl, dtype=torch.float64, device=self.device)
        self.b_r_torch = torch.tensor(self.b_r, dtype=torch.float64, device=self.device)
        self.b_p_torch = torch.tensor(self.b_p, dtype=torch.float64, device=self.device)
        self.b_eq_torch = torch.tensor(self.b_eq, dtype=torch.float64, device=self.device)
        
        self.G = torch.vstack([
            torch.tensor(self.A_p.toarray(), dtype=torch.float64, device=self.device),
            torch.tensor(self.A_r.toarray(), dtype=torch.float64, device=self.device)
        ])

        self.h = torch.vstack([
            self.b_p_torch,
            self.b_r_torch
        ]).squeeze()
        
        self.A = torch.tensor(self.A_fl.toarray(), dtype=torch.float64, device=self.device)
        self.b = self.b_fl_torch.squeeze()

    def flow_constraint_matrix(self, X_flat):

        violation = torch.sparse.mm(self.A_fl_torch, X_flat.unsqueeze(1)).squeeze() - self.b_fl_torch.reshape((-1,))

        assert len(violation.shape)==1

        # return self.A_fl_torch@X_flat - self.b_fl_torch
        return violation

    def reachability_constraint_matrix(self, X_flat):

        return torch.sparse.mm(self.A_r_torch, X_flat.unsqueeze(1)).squeeze() - self.b_r_torch
    
    
    def targeted_deception(self, target_occupancy_measures, initvals, beta = 0):
        """
        Solve MMDP with targeted deception (Optimization Problem 5)
        """

        beta = -beta
        
        assert beta < 0

        print("Solving Targeted Deception with QPTH on", self.device)
        
        time0 = time.time()

        # Objective function
        n = self.n_states * self.n_actions  # Size of the matrix

        diagonal_values = np.full(n, -2 * beta, dtype=np.float64)  # float64 diagonal values
        P = sp.diags(diagonal_values, format="csc").astype(np.float64)

        q = ((2 * beta) * target_occupancy_measures.flatten() - self.r.flatten()).astype(np.float64)
        
        P_dense = P.toarray()  # Convert sparse to dense NumPy array
        P_torch = torch.tensor(P_dense, dtype=torch.float64).cuda() 
        q_torch = torch.tensor(q, dtype=torch.float64).cuda()

        # Solve QP problem
        
        sol = QPFunction()(P_torch, q_torch, self.G, self.h, self.A, self.b).squeeze()
        
        print(sol.shape)
        
        print(sol)
        
        print("Time :", time.time()-time0)

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

        # Convert to tensor on GPU
        # occupancy_measure = torch.tensor(occupancy_measure, dtype=torch.float64)
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
        value_function = self.policy_evaluation(self.transitions_torch, self.r_torch, policy, 1e-5, self.gamma)

        # Revenue (r must be a tensor on the same device)
        revenue = torch.sum(self.r_torch * occupancy_measures).item()
        
        print(revenue)
        
        print("Time for evaluating solution:", time.time()-time0)

        return occupancy_measures, policy, value_function, revenue