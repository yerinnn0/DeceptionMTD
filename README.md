## Files

- `MDP.py` includes settings and functions for multi-agent MDP.
- `policy_optimization_cvxopt.py` includes settings and functions for solving MMDP as optimization problem.
- `irl_maxent_rsa.py` includes settings and functions for generating trajectory and solving IRL.
- `execute_cvxopt.py` includes codes for parameter setting and execution.

## Execution

You can execute the whole simulation by running `execute_cvxopt.py`.

## Change Settings

1. At the end of `execute_cvxopt.py`, you can change the following parameters.

- **$\beta$** (Deception parameter): `beta_vec = np.arange(0,0.3,0.03)`
- **N_agents** (Number of agents): `N_agents = 3`
- **target_agents** (List of index of true target among N agents): `target_agents = [0]`
- **decoy_agents** (List of index of decoy target which we want to mislead adversary to(Used in targeted deception and equivocal deception)): `decoy_agents = [1,2]`

2. At the end of the function `run_all` in `execute_cvxopt.py`, you can add or change the format of saving the result.
