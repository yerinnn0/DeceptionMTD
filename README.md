# DeceptionMTD

This repository implements occupancy-measure optimization for diversionary,
targeted, and equivocal deception in Markov decision processes. The main
application is a multi-agent moving-target-defense (MTD) problem in which a
defender selects deceptive policies while maintaining a required level of
system performance. The repository also includes a single-agent stochastic
gridworld as an additional example outside the MTD setting.

## Repository structure

- `MDP.py`: Original multi-agent MTD environment, including transition and
  reward construction.
- `GridworldMDP.py`: Additional single-agent stochastic gridworld environment.
- `config.py`: MDP, deception, solver, IRL, and output settings for both
  environments.
- `execute_serial.py`: Experiment workflow for nominal optimization,
  deceptive-policy optimization, optional IRL, and result storage.
- `policy_optimization/`: Optimization implementations for SciPy, OSQP,
  Pyomo, Gurobi, and other supported solvers.
- `irl/`: Maximum-entropy, deep maximum-entropy, and apprenticeship-learning
  IRL implementations.
- `plot.ipynb`: Analysis and plotting notebook for the main MTD experiments.
- `plot_trajectory.py`: State-occupancy heatmaps for the additional gridworld
  example.
- `plot_revenue_bound.py`: Gridworld revenue and theoretical-bound plots for 
  the additional gridworld example.

## Installation

The code requires Python 3. The common numerical and plotting dependencies can
be installed with:

```bash
pip install numpy scipy torch osqp matplotlib tqdm
```

The main MTD experiments may additionally require the solver selected in
`config.py`, such as Pyomo or Gurobi. A valid Gurobi installation and license
are required when `gurobi` is selected.

## Main experiment: Multi-agent MTD

The original MTD environment models multiple agents with local cyber states
and actions. Agents can represent the real system components whose operation
must be protected and decoy components used to mislead an observer. The joint
state and action spaces are constructed from the local spaces of all agents.

### Experiment workflow

Running `execute_serial.py` performs the following workflow:

1. Construct the multi-agent MDP from the settings in `config.py`.
2. Solve the nominal MDP and obtain its occupancy measure, policy, value
   function, and revenue.
3. Solve the selected diversionary, targeted, or equivocal deception problem
   over the specified beta values.
4. Optionally generate trajectories and estimate rewards using IRL.
5. Save intermediate and consolidated results as PKL files.

### Selecting the MTD environment

Set the following value near the top of `config.py`:

```python
environment_name = "mtd"
```

The principal MTD settings include:

```python
N_agents = 5
n_local_states = 4
n_local_actions = 3
local_initial_state = 0
local_goal_state = 0

real_agents = [0]
target_decoy_agents = [1]
target_occupancy_measure_values = [3, 1, -1, -3]
```

The following dictionaries record the settings saved with every result:

- `MMDP_SETTINGS`: number of agents, local state/action dimensions, initial
  and goal states, discount factor, and task-reachability requirement.
- `DECEPTION_SETTINGS`: deception type, beta values, real and target-decoy
  agents, and targeted-deception occupancy weights.
- `IRL_SETTINGS`: whether IRL is run and its model, learning, trajectory, and
  iteration settings.
- `OPTIMIZATION_SETTINGS`: solver selected for the nominal and three
  deceptive optimization problems.

Choose the deception type and beta range in the deception section of
`config.py`:

```python
deception_type = "diversionary"  # or "targeted", "equivocal"
beta_vec = ...
```

The default MTD solver mapping uses Pyomo for the nominal, diversionary, and
targeted problems and Gurobi for equivocal deception. These choices can be
changed through `optimization_solver` in `config.py`.

### Running the MTD experiment

From the repository root, run:

```bash
python execute_serial.py
```

Set `RUN_IRL = True` to continue from deceptive-policy optimization to
trajectory generation and reward estimation. Set it to `False` when only the
nominal and deceptive optimization results are needed. The main MTD results
can be analyzed with `plot.ipynb` after updating its experiment file names to
match the current `EXPERIMENT_NAME` and number of agents.

Output names are generated from the environment, number of agents, experiment
name, deception type, and IRL model. Existing consolidated files are loaded
instead of recomputed. To rerun an experiment from scratch, use a new
`EXPERIMENT_NAME` or move the existing result files to another directory.

## Additional example: Stochastic gridworld

The gridworld example applies the same deceptive-policy optimization framework
to a structurally different, non-MTD navigation problem. It provides an
intuitive comparison of the three deception types and illustrates the
trade-off between deceptive behavior and nominal revenue.

### Gridworld setting

The additional example is a continuing single-agent MDP with the following
settings:

- Grid: `5 x 5` with row-major state indices.
- Start: `(0, 0)`.
- Terminal: `(4, 4)`, followed by a deterministic reset to the start.
- Actions: `up`, `right`, `down`, and `left`.
- Discount factor: `gamma = 0.9`.
- Transition model: the selected action is executed with probability `0.9`,
  while probability `0.1` is distributed uniformly over all four actions.
  Thus, the intended action has total probability `0.925` and each other
  action has probability `0.025`.
- Boundary collision: an action that would leave the grid keeps the agent in
  its current state.

Rewards follow the current-state convention `r(s, a)` and are identical across
actions at a given state. Every action has movement reward `-0.1`, a state on
the preferred upper/right path receives an additional `+0.2`, and the terminal
receives an additional `+10.0`. The resulting rewards are `-0.1` at regular
states (including the start), `0.1` at non-terminal preferred-path states, and
`10.1` at the terminal.

The task set follows the upper and right boundaries, excludes the start, and
includes the terminal. The decoy route follows the left and lower boundaries
and excludes the shared start and terminal. The task constraint uses:

```python
v_reach = 0.3 / (1 - gamma)  # 3.0 when gamma = 0.9
```

The total discounted occupancy mass is `1 / (1 - gamma) = 10`; therefore,
this setting requires transition mass into the preferred task states of at
least `3.0`. The terminal is excluded from the preferred-versus-decoy
comparison in equivocal deception. When `x_tar` is `None`, targeted deception
uses a flow-feasible occupancy target induced by the lower/left decoy-route
policy.

### Running the gridworld experiment

Select the additional environment and disable IRL for the revenue experiment:

```python
environment_name = "gridworld"
RUN_IRL = False
```

Run each deception type separately. The final experiment uses 11 beta values,
including zero and the endpoint:

```python
# Diversionary
deception_type = "diversionary"
beta_vec = np.linspace(0.0, 0.07, 11)

# Targeted
deception_type = "targeted"
beta_vec = np.linspace(0.0, 0.15, 11)

# Equivocal
deception_type = "equivocal"
beta_vec = np.linspace(0.0, 0.05, 11)
```

After selecting one block, run:

```bash
python execute_serial.py
```

With the default experiment name, the consolidated gridworld files are:

```text
result_gridworld_5x5_1_lp.pkl
final_rev1_gridworld_5x5_1_div_opt.pkl
final_rev1_gridworld_5x5_1_tar_opt.pkl
final_rev1_gridworld_5x5_1_equ_opt.pkl
```

When `SAVE_INTERMEDIATE_RESULT = True`, a separate PKL is also saved for every
beta value. The endpoint solutions retain approximately the following
percentages of nominal revenue:

| Deception type | Beta | Revenue (% of nominal) |
|---|---:|---:|
| Diversionary | 0.07 | 84.62% |
| Targeted | 0.15 | 84.51% |
| Equivocal | 0.05 | 90.85% |

### Plotting the gridworld results

After generating the nominal and all three deceptive results, create the
1-by-4 state-occupancy comparison with:

```bash
python plot_trajectory.py
```

This writes `gridworld_occupancy_1x4.png`. The script reads the endpoint
intermediate PKLs specified in its `PKL_FILES` dictionary.

Create the three revenue and theoretical-bound panels with:

```bash
python plot_revenue_bound.py
```

This writes `gridworld_5x5_revenue_bounds.png` and prints the beta values,
actual revenue, revenue percentage, theoretical bound, and task-reachability
diagnostics for each deception type. If `EXPERIMENT_NAME` is changed, update
the `PKL_FILES` dictionaries at the top of the plotting scripts.
