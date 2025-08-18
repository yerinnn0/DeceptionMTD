import numpy as np


SAVE_INTERMEDIATE_RESULT = True
EXPERIMENT_NAME = "irl_average2"


### MMDP Settings
N_agents = 4
n_local_states = 4
n_local_actions = 3
local_initial_state = 0
local_goal_state = 0
gamma = 0.9
v_reach = 0.7/(1-gamma)
feature_map = "identity"
build_transition_matrix = True

### IRL Settings
irl_model = 'apprenticeship'
irl_repetitions = np.arange(49,50,1)

if irl_model == "maxent":
    irl_epoch = 20
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
    irl_num_traj = 100
    irl_len_traj = 100
    irl_max_iter = 300


### Deception Settings
target_occupancy_measure_values = [3,1,-1,-3]
deception_type = 'equivocal'
start_beta = 0
end_beta = 1.1
beta_step = 0.1
beta_vec = np.arange(start_beta, end_beta, beta_step)
real_agents = [0]
target_decoy_agents = [1]

### Optimization Settings
optimization_solver = {'mdp': 'gurobi', 'diversionary':'pyomo', 'targeted':'gurobi', 'equivocal':'gurobi'}

### MMDP Settings
MMDP_SETTINGS = {
    "N_agents" : N_agents,
    "n_local_states" : n_local_states,
    "n_local_actions" : n_local_actions,
    "local_initial_state" : local_initial_state,
    "local_goal_state" : local_goal_state,
    "gamma" : gamma,
    "v_reach" : v_reach
}

### IRL Settings
IRL_SETTINGS = {
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
    "target_decoy_agents" : target_decoy_agents
}

### Optimization Settings
OPTIMIZATION_SETTINGS = {
    "optimization_solver" : optimization_solver
}


### File Name
save_file_format = {
    'all': 'result_'+str(N_agents)+'_all.pkl',
    'lp': 'result_' +str(N_agents)+'_lp.pkl',
    'deception': EXPERIMENT_NAME + '_'+str(N_agents)+'_'+deception_type[:3]+'_opt.pkl',
    'irl': EXPERIMENT_NAME + '_'+str(N_agents)+'_'+deception_type[:3]+'_'+ irl_model+'.pkl',
}
