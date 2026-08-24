"""
Solve optimization problem for deceptive policy
"""
import numpy as np
from scipy.optimize import minimize
import time

from .policy_optimization import PolicyOptimization


def _as_numpy_matrix(value, shape, name):
    """Convert NumPy/Torch occupancy inputs at the SciPy solver boundary."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value, dtype=np.float64)
    if array.size != int(np.prod(shape)):
        raise ValueError(f"{name} must contain {int(np.prod(shape))} values")
    return array.reshape(shape)


class PolicyOptimizationScipy(PolicyOptimization):

    def __init__(self, mmdp):
        super().__init__(mmdp)

    def solve_MDP(self):
        """
        Solve MMDP without deception (Optimization Problem 3)
        """

        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        r = self.mmdp.joint_rewards

        def MDP_objective(X_flat):
            X = X_flat.reshape((n_states, n_actions))
            return -np.sum(r * X)

        initial_guess = np.ones(n_states*n_actions)
        constraints = [{'type': 'eq', 'fun': self.flow_constraint},{'type': 'ineq', 'fun': self.reachability_constraint}]

        print("Solving LP...")
        time0 = time.time()

        result = minimize(MDP_objective, initial_guess, constraints=constraints, bounds=[(0, None) for _ in range(n_states * n_actions)], options={'disp': False})

        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        sol = result.x
        
        return self.evaluation(sol)
    
    def diversionary_deception(self, occupancy_measures, init = None, beta = 1):
        """
        Solve MMDP with diversionary deception (Optimization Problem 4)
        """
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        beta = beta
        shape = (n_states, n_actions)
        r = np.asarray(self.mmdp.joint_rewards, dtype=np.float64).reshape(shape)
        occupancy_measures = _as_numpy_matrix(
            occupancy_measures, shape, "occupancy_measures"
        )

        def diversionary_objective(X_flat):
            X = X_flat.reshape((n_states, n_actions))
            return -np.sum(beta * X**2 + (r - 2 * beta * occupancy_measures) * X)
        
        initial_guess = (
            np.ones(n_states * n_actions)
            if init is None
            else _as_numpy_matrix(init, shape, "init").reshape(-1)
        )
        
        constraints = [{'type': 'eq', 'fun': self.flow_constraint}, {'type': 'ineq', 'fun': self.reachability_constraint}]

        print("Solving Diversionary Deception...")
        
        time0 = time.time()
        result = minimize(diversionary_objective, initial_guess, constraints=constraints, bounds=[(0, None) for _ in range(n_states * n_actions)], options={'disp': False})

        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        sol = result.x

        return self.evaluation(sol)
    
    def targeted_deception(self, target_occupancy_measures, initvals=None, beta = 1):
        """
        Solve MMDP with targeted deception (Optimization Problem 5)
        """

        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        beta = -beta
        shape = (n_states, n_actions)
        r = np.asarray(self.mmdp.joint_rewards, dtype=np.float64).reshape(shape)
        target_occupancy_measures = _as_numpy_matrix(
            target_occupancy_measures, shape, "target_occupancy_measures"
        )

        def targeted_objective(X_flat):
            X = X_flat.reshape((n_states, n_actions))

            return -np.sum(beta * X**2 + (r - 2 * beta * target_occupancy_measures) * X)
        
        initial_guess = (
            np.ones(n_states * n_actions)
            if initvals is None
            else _as_numpy_matrix(initvals, shape, "initvals").reshape(-1)
        )
        
        constraints = [{'type': 'eq', 'fun': self.flow_constraint}, {'type': 'ineq', 'fun': self.reachability_constraint}]

        print("Solving Targeted Deception...")
        
        time0 = time.time()
        result = minimize(targeted_objective, initial_guess, constraints=constraints, bounds=[(0, None) for _ in range(n_states * n_actions)], options={'disp': False})

        print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

        sol = result.x

        return self.evaluation(sol)
        
    # def equivocal_deception(self):
    #     """
    #     Solve MMDP with equivocal deception (Optimization Problem 6)
    #     """
    #     n_states = self.mmdp.n_joint_states
    #     n_actions = self.mmdp.n_joint_actions
    #     r = self.mmdp.joint_rewards

    #     def equivocal_objective(X_flat):
    #         X = X_flat.reshape((n_states, n_actions))
    #         return -np.sum(r * X)
        
    #     initial_guess = np.ones(n_states*n_actions)
        
    #     constraints = [{'type': 'eq', 'fun': self.flow_constraint}, 
    #                    {'type': 'ineq', 'fun': self.reachability_constraint}, 
    #                    {'type': 'eq', 'fun': self.equivocal_constraint}]

    #     print("Solving Equivocal Deception...")
    #     time0 = time.time()
    #     result = minimize(equivocal_objective, initial_guess, constraints=constraints, bounds=[(0, None) for _ in range(n_states * n_actions)], options={'disp': False})

    #     print("Time :", time.time()-time0, "Time per state :", (time.time()-time0)/n_states)

    #     sol = result.x

    #     return self.evaluation(sol)
