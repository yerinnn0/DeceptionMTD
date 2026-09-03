import numpy as np


SAVE_INTERMEDIATE_RESULT = True
EXPERIMENT_NAME = "final_rev1"
RUN_IRL = False


### Environment / MMDP Settings
# "mtd": original multi-agent cyber MDP
# "gridworld": single-agent 5x5 route-deception example
environment_name = "gridworld"

gamma = 0.9
feature_map = "identity"
build_transition_matrix = True

if environment_name == "mtd":
    N_agents = 5
    n_local_states = 4
    n_local_actions = 3
    local_initial_state = 0
    local_goal_state = 0
    v_reach = 0.7/(1-gamma)

    real_agents = [0]
    target_decoy_agents = [1]
    target_occupancy_measure_values = [3, 1, -1, -3]
    path_reward = None
    terminal_reward = None
    movement_cost = None
    x_tar = None

elif environment_name == "gridworld":
    grid_shape = (5, 5)
    grid_rows, grid_cols = grid_shape
    grid_start_coord = (0, 0)
    grid_terminal_coord = (grid_rows - 1, grid_cols - 1)

    N_agents = 1
    n_local_states = grid_rows * grid_cols
    n_local_actions = 4
    local_initial_state = 0
    local_goal_state = n_local_states - 1
    v_reach = 0.3/(1-gamma)

    # Goal states (preferred path) follows the upper/right boundary and includes terminal T.
    grid_goal_states = (
        list(range(1, grid_cols))
        + [row * grid_cols + (grid_cols - 1) for row in range(1, grid_rows)]
    )

    path_reward = 0.2
    terminal_reward = 10.0
    movement_cost = -0.1
    x_tar = None  # None => flow-feasible target induced by the decoy-route policy

    real_agents = [0]
    target_decoy_agents = []
    target_occupancy_measure_values = None

else:
    raise ValueError("environment_name must be 'mtd' or 'gridworld'")

### IRL Settings
irl_model = 'maxent'
irl_repetitions = np.arange(75,0,-1)

if irl_model == "maxent":
    irl_epoch = 50
    irl_learning_rate = 0.5
    irl_layers = None
    irl_num_traj = 500
    irl_len_traj = 500
    irl_max_iter = None
elif irl_model == "deep_maxent":
    irl_epoch = 50
    irl_learning_rate = 0.1
    irl_layers = (64,32)
    irl_num_traj = 500
    irl_len_traj = 500
    irl_max_iter = None
elif irl_model == "apprenticeship":
    irl_epoch = None
    irl_learning_rate = None
    irl_layers = None
    irl_num_traj = 500
    irl_len_traj = 500
    irl_max_iter = 500


### Deception Settings
deception_type = 'equivocal'
start_beta = 0
end_beta = 0.051
beta_step = 0.005
beta_vec = np.arange(start_beta, end_beta, beta_step)

### Optimization Settings
"""
Recommended solvers:
    'mdp', 'diversionary', 'targeted' : 'pyomo',
    'equivocal' : 'gurobi'
"""
if environment_name == "gridworld":
    optimization_solver = {
        'mdp': 'osqp',
        'diversionary': 'scipy',
        'targeted': 'scipy',
        'equivocal': 'osqp',
    }
else:
    optimization_solver = {
        'mdp': 'pyomo',
        'diversionary': 'pyomo',
        'targeted': 'pyomo',
        'equivocal': 'gurobi',
    }

### MMDP Settings
MMDP_SETTINGS = {
    "environment_name": environment_name,
    "N_agents" : N_agents,
    "n_local_states" : n_local_states,
    "n_local_actions" : n_local_actions,
    "local_initial_state" : local_initial_state,
    "local_goal_state" : local_goal_state,
    "gamma" : gamma,
    "v_reach" : v_reach
}
if environment_name == "gridworld":
    MMDP_SETTINGS.update({
        "grid_shape": grid_shape,
        "start_coord": grid_start_coord,
        "terminal_coord": grid_terminal_coord,
        "start_state": local_initial_state,
        "terminal_state": local_goal_state,
        "task_states": grid_goal_states,
        "task_constraint": "preferred_goal_path_reachability",
        "path_reward": path_reward,
        "terminal_reward": terminal_reward,
        "movement_cost": movement_cost,
    })

### IRL Settings
IRL_SETTINGS = {
    "run_irl": RUN_IRL,
    "irl_model": irl_model,
    "irl_epoch" : irl_epoch,
    "irl_learning_rate" :irl_learning_rate,
    "irl_num_traj" : irl_num_traj,
    "irl_len_traj" : irl_len_traj
}

### Deception Settings
DECEPTION_SETTINGS = {
    "target_occupancy_measure_values" : target_occupancy_measure_values,
    "deception_type" : deception_type,
    "beta_vec" : beta_vec,
    "real_agents" : real_agents,
    "target_decoy_agents" : target_decoy_agents,
    "x_tar_source": (
        "decoy_policy" if environment_name == "gridworld" and x_tar is None
        else "user" if environment_name == "gridworld"
        else "legacy_state_weights"
    ),
}

### Optimization Settings
OPTIMIZATION_SETTINGS = {
    "optimization_solver" : optimization_solver
}


### File Name
environment_file_tag = (
    str(N_agents)
    if environment_name == "mtd"
    else environment_name + '_' + str(grid_rows) + 'x' + str(grid_cols) + '_' + str(N_agents)
)
save_file_format = {
    'all': 'result_'+environment_file_tag+'_all.pkl',
    'lp': 'result_'+environment_file_tag+'_lp.pkl',
    'deception': EXPERIMENT_NAME+'_'+environment_file_tag+'_'+deception_type[:3]+'_opt.pkl',
    'irl': EXPERIMENT_NAME+'_'+environment_file_tag+'_'+deception_type[:3]+'_'+irl_model+'.pkl',
}
