import numpy as np

### MDP Settings
n_local_states = 4
n_local_actions = 3
local_initial_state = 0
local_goal_state = 0
gamma = 0.9
v_reach = 0.7/(1-gamma)

### MMDP Settings
N_agents = 2

### IRL Settings
irl_epoch = 6
irl_learning_rate = 0.9
irl_num_traj = 100
irl_len_traj = 100

### Deception Settings
target_occupancy_measure_values = [3,1,-1,-3]
deception_type = 'diversionary'
beta_vec = np.arange(0,0.1,0.01)
real_agents = [0]
target_decoy_agents = [1]

### File Name
simulation_type = 'all'
save_file_format = {
    'all': 'result_'+str(N_agents)+'_all.pkl',
    'lp': 'result_'+str(N_agents)+'_lp.pkl',
    'deception': 'result_'+str(N_agents)+'_'+deception_type[:3]+'_opt.pkl',
    'irl': 'result_'+str(N_agents)+'_'+deception_type[:3]+'_irl.pkl',
}
save_file_name = save_file_format[simulation_type]
