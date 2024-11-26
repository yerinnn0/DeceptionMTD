import numpy as np
from itertools import product

from MDP import create_rewards, create_decoy_rewards, MultiAgentGridworld
from policy_optimization_cvxopt import PolicyOptimizationCVX
from irl_maxent_rsa import IRL

import multiprocessing as mp
import pickle
import os

def run_all(beta, num_traj =100, len_traj=100, type = "diversionary", v_reach =8):
    
    def calculate_error(true_value_func, estimated_value_func, scale = None):
        """
        Calculate the L1 norm of difference in value function
        
        """
        if scale == None:
            scale = np.sum(np.abs(true_value_func))

        return np.sum(np.abs(true_value_func - estimated_value_func))/scale
    
    def build_multi_agent_rewards(n_states, n_local_actions, N_agents, target_agents = [0]):
        """
        Create multi-agent rewards
        
        return: Reward matrix with shape (n_states, n_local_actions, N_agents)
        """
        rewards = np.zeros([n_states, n_local_actions, N_agents])
        
        for agent_id in range(N_agents):
            if agent_id in target_agents:
                rewards[:, :, agent_id] = create_rewards()
            else:
                rewards[:, :, agent_id] = create_decoy_rewards()
            
        return rewards
    
    def determine_goal_decoy_states(n_states, N_agents, target_agents = [0], decoy_agents = [1]):
        """
        Determine goal states and decoy states
        Goal states: When target agents are in "N(normal)" state
        Decoy states: When decoy agents are in "N(normal)" state 
        
        return: 1d array of goal states, 1d array of decoy states
        """
        
        if N_agents == 2:
            goal_states = np.array([0,1,2,3])
            decoy_states = np.array([0,4,8,12])
        elif N_agents == 3:
            goal_states = range(0,n_states**(N_agents-1))
            decoy_states = []
            for i in range(n_states):
                decoy_states.append(np.arange(n_states**(N_agents-1)*i, n_states**(N_agents-1)*i + n_states))
            decoy_states = np.array(decoy_states).reshape(-1,)  
        elif N_agents == 1:
            goal_states = np.array([0])
            decoy_states = np.array([])   
        
        return goal_states, decoy_states
    
    def build_target_occupancy_measure(mmdp, n_states, n_local_actions, N_agents, target_agents = [1]):
        """
        Set target occupancy measure
        
        return: Target occupancy measure as array with shape (mmdp.n_joint_states,mmdp.n_joint_actions)
        """
        
        target_occupancy_measure = np.zeros((mmdp.n_joint_states,mmdp.n_joint_actions))
        if N_agents ==2:
            for i, j, k, a in product(range(n_states), range(n_states), range(n_local_actions), range(n_local_actions)):
                if j == 0:
                    target_occupancy_measure[i*n_states +j, k*n_local_actions + a] = 3
                elif j == 1:
                    target_occupancy_measure[i*n_states +j, k*n_local_actions + a] = 1
                elif j == 2:
                    target_occupancy_measure[i*n_states +j, k*n_local_actions + a] = -1
                elif j == 3:
                    target_occupancy_measure[i*n_states +j, k*n_local_actions + a] = -3
        elif N_agents ==3:
            for i, j, k, a, b, c in product(range(n_states), range(n_states), range(n_states), range(n_local_actions), range(n_local_actions), range(n_local_actions)):
                if j == 0:
                    target_occupancy_measure[i*(n_states**2) +j*n_states +k, a*(n_local_actions**2) + b*n_local_actions +c] = 2
                elif j == 1:
                    target_occupancy_measure[i*(n_states**2) +j*n_states +k, a*(n_local_actions**2) + b*n_local_actions +c] = 1
                elif j == 2:
                    target_occupancy_measure[i*(n_states**2) +j*n_states +k, a*(n_local_actions**2) + b*n_local_actions +c] = -1
                elif j == 3:
                    target_occupancy_measure[i*(n_states**2) +j*n_states +k, a*(n_local_actions**2) + b*n_local_actions +c] = -3
        
        return target_occupancy_measure

        
    # MMDP Settings
    n_states = 4
    N_agents = 3
    n_local_actions = 3
    gamma = 0.9
    
    target_agents = [0]

    # Deterministic Initial States
    initial_state = 0
    initial_distribution = np.zeros((n_states, N_agents))
    initial_distribution[initial_state] = 1
    initial_distribution = initial_distribution.T.reshape(-1,1)
    
    # Rewards
    rewards = build_multi_agent_rewards(n_states, n_local_actions, N_agents, target_agents)
       
    # Deterministic goal states
    goal_states, decoy_states = determine_goal_decoy_states(n_states, N_agents, target_agents)
    
    mmdp = MultiAgentGridworld(N_agents, initial_distribution, n_states, n_local_actions, rewards, gamma, v_reach = v_reach)
    mmdp.set_goal_states(goal_states, decoy_states)
    
    policy_opt = PolicyOptimizationCVX(mmdp)
    irl = IRL(mmdp, epochs =7, learning_rate = 0.9)
    
    # Target occupancy measure
    target_occupancy_measure = build_target_occupancy_measure(mmdp, n_states, n_local_actions, N_agents)
    
    ## 1. Solve LP
    occupancy_measures, policy, value_function, revenue = policy_opt.solve_lp_cvxopt()

    ## 2. Solve QP
    if type == "diversionary":
        deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue = \
                    policy_opt.diversionary_deception(occupancy_measures, beta = beta)
    elif type == "targeted":
        deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue = \
                    policy_opt.targeted_deception(target_occupancy_measure, beta = beta)
    elif type == "equivocal":
        deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue = \
                    policy_opt.equivocal_deception()
    else:
        print ("Undefined Deception Type")
        
    assert(deceptive_occupancy_measures >= 0).all()
    assert(deceptive_policy >= 0).all()

    ## 3. Solve IRL
    print("Solving IRL...")
    trajectories = irl.generate_trajectories(deceptive_policy, num_traj, len_traj)
    estimated_value_function = irl.estimate_value_function(np.expand_dims(trajectories, axis = 0))
        
    scale = np.sum(np.abs(value_function))
    deception_error = calculate_error(value_function, deceptive_value_function, scale)
    inverse_learning_error = calculate_error(deceptive_value_function, estimated_value_function, scale)
    total_estimation_error = calculate_error(value_function, estimated_value_function, scale)
    
    policy_difference = np.sum(np.abs(policy - deceptive_policy))
    
    result = {
        'error' : {
            'deception_error': deception_error, 
            'inverse_learning_error': inverse_learning_error, 
            'total_estimation_error': total_estimation_error
        },
        'value_function':{
            'value_function': value_function,
            'deceptive_value_function': deceptive_value_function,
            'estimated_value_function': estimated_value_function,
            'self_estimated_value_function': irl.self_estimated_value_function
        },
        'policy':{
          'policy': policy,
          'deceptive_policy': deceptive_policy,
          'estimated_policy': irl.estimated_policy  
        },
        'revenue':{
            'revenue': revenue,
            'deceptive_revenue':deceptive_revenue
        },
        'reward':{
            'reward': mmdp.joint_rewards,
            'estimated_reward': irl.expected_rewards
        },
        'occupancy_measure':{
            'occupancy_measure':occupancy_measures,
            'deceptive_occupancy_measure': deceptive_occupancy_measures,
            'target_occupancy_measure': target_occupancy_measure,
        },
        'policy_difference': np.array(policy_difference),
        'observed_trajectory': trajectories
    }

    return result

beta_vec = np.arange(0,20,2)

if __name__ == '__main__':
    with mp.Pool() as pool:
        
        diversionary_results = pool.starmap(run_all, [(beta_vec[i],100,100, "diversionary",7) for i in range(len(beta_vec))])
        targeted_results = pool.starmap(run_all, [(beta_vec[i],100,100, "targeted",7) for i in range(len(beta_vec))])
        equivocal_results = pool.starmap(run_all, [(beta_vec[i],100,100, "equivocal",7) for i in range(len(beta_vec))])
        
        experiment_logger = {
            'beta_vec': beta_vec,
            'diversionary_results' : diversionary_results, 
            'targeted_results' : targeted_results, 
            'equivocal_results' : equivocal_results
        }
        
        save_file_name = 'result.pkl'
        save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name))
        with open(save_str, 'wb') as f:
            pickle.dump(experiment_logger, f, protocol=pickle.HIGHEST_PROTOCOL)
      

