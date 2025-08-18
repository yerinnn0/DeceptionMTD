## Files

- `MDP.py` includes settings and functions for multi-agent MDP.
- `policy_optimization` includes settings and functions for solving MMDP as optimization problem.
- `IRL` includes settings and functions for generating trajectory and solving IRL.
- `execute_serial.py` includes codes for parameter setting and execution.
  
      1. Set up the problem
  
      2. Solve the original MDP
  
      3. Solve the MDP with deception
  
      4. Solve IRL (Generates trajectory -> Estimates rewards)
  
      5. Save the results

## Execution

You can execute the whole simulation by running `execute_serial.py`.

## Change Settings

At `config.py`, you can change the following parameters.
  - MMDP_SETTINGS = {
      "N_agents", "n_local_states",  "n_local_actions", "local_initial_state", "local_goal_state", "gamma", "v_reach"
  }
  - IRL_SETTINGS = {
      "irl_model", "irl_epoch", "irl_learning_rate", "irl_num_traj", "irl_len_traj", "irl_max_iter"
  }
  - DECEPTION_SETTINGS = {
      "target_occupancy_measure_values", "deception_type", "beta_vec", "real_agents", "target_decoy_agents"
  }
  - OPTIMIZATION_SETTINGS = {
      "optimization_solver"
  }

