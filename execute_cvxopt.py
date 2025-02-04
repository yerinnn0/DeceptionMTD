import numpy as np
from itertools import product

from MDP import create_rewards, create_decoy_rewards, MultiAgentGridworld
from policy_optimization_cvxopt import PolicyOptimizationCVX
from irl_maxent_rsa import IRL
from config import *

import multiprocessing as mp
import pickle
import os

def run_all(beta):
    
    def calculate_error(true_value_func, estimated_value_func, scale = None):
        """
        Calculate the L1 norm of difference in value function
        
        """
        if scale == None:
            scale = np.sum(np.abs(true_value_func))

        return np.sum(np.abs(true_value_func - estimated_value_func))/scale
    
    def build_multi_agent_rewards(n_states, n_local_actions, N_agents, real_agents = [0]):
        """
        Create multi-agent rewards
        
        return: Reward matrix with shape (n_states, n_local_actions, N_agents)
        """
        rewards = np.zeros([n_states, n_local_actions, N_agents])
        
        for agent_id in range(N_agents):
            if agent_id in real_agents:
                rewards[:, :, agent_id] = create_rewards()
            else:
                rewards[:, :, agent_id] = create_decoy_rewards()
            
        return rewards
    
    def find_the_joint_state(n_states, N_agents, agent, state):
        
        shape = (n_states,)*N_agents
        joint_states = np.zeros(shape)
        
        slices = [slice(None)] * N_agents
        slices[agent] = state  # Set the i-th dimension index to 0
        joint_states[tuple(slices)] = 1  # Fill the corresponding entries with 1
        
        joint_states = joint_states.flatten()
        joint_states_idx = np.where(joint_states==1)[0]
        
        return joint_states_idx
    
    def determine_goal_decoy_states(n_local_states, N_agents, real_agents = [0], target_decoy_agents = [1]):
        """
        Determine goal states and decoy states
        Goal states: When target agents are in "N(normal)" state
        Decoy states: When decoy agents are in "N(normal)" state 
        
        return: 1d array of goal states, 1d array of decoy states
        """
        goal_states = []
        decoy_states = []
        
        for i in real_agents:
            goal_states.extend(find_the_joint_state(n_local_states, N_agents, i, local_goal_state))
        for i in target_decoy_agents:
            decoy_states.extend(find_the_joint_state(n_local_states, N_agents, i, local_goal_state))
        
        return np.array(goal_states), np.array(decoy_states)
    
    def build_target_occupancy_measure(n_local_states, n_local_actions, N_agents, target_decoy_agents = [1]):
        """
        Set target occupancy measure
        
        return: Target occupancy measure as array with shape (mmdp.n_joint_states,mmdp.n_joint_actions)
        """
        target_occupancy_measure = np.zeros((n_local_states**N_agents, n_local_actions**N_agents), dtype=int)
        
        for i in target_decoy_agents:
            state0 = find_the_joint_state(n_local_states, N_agents, i, 0)
            state1 = find_the_joint_state(n_local_states, N_agents, i, 1)
            state2 = find_the_joint_state(n_local_states, N_agents, i, 2)
            state3 = find_the_joint_state(n_local_states, N_agents, i, 3)
            
            target_occupancy_measure[state0,:] += target_occupancy_measure_values[0]
            target_occupancy_measure[state1,:] += target_occupancy_measure_values[1]
            target_occupancy_measure[state2,:] += target_occupancy_measure_values[2]
            target_occupancy_measure[state3,:] += target_occupancy_measure_values[3]
            
        return target_occupancy_measure


    # Deterministic Initial States
    initial_distribution = np.zeros((n_local_states, N_agents))
    initial_distribution[local_initial_state] = 1
    initial_distribution = initial_distribution.T.reshape(-1,1)
    
    # Rewards
    rewards = build_multi_agent_rewards(n_local_states, n_local_actions, N_agents, real_agents)
       
    # Deterministic goal states
    goal_states, decoy_states = determine_goal_decoy_states(n_local_states, N_agents, real_agents, target_decoy_agents)
    
    # Target occupancy measure
    target_occupancy_measure = build_target_occupancy_measure(n_local_states, n_local_actions, N_agents, target_decoy_agents)
    
    mmdp = MultiAgentGridworld(N_agents, initial_distribution, n_local_states, n_local_actions, rewards, gamma, v_reach)
    mmdp.set_goal_states(goal_states, decoy_states)
    
    policy_opt = PolicyOptimizationCVX(mmdp)
    irl = IRL(mmdp, epochs = irl_epoch, learning_rate = irl_learning_rate)
    
    # 1. Solve LP
    occupancy_measures, policy, value_function, revenue = policy_opt.solve_lp_cvxopt()

    ## 2. Solve QP
    if deception_type == "diversionary":
        deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue = \
                    policy_opt.diversionary_deception(occupancy_measures, beta = beta)
    elif deception_type == "targeted":
        deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue = \
                    policy_opt.targeted_deception(target_occupancy_measure, beta = beta)
    elif deception_type == "equivocal":
        deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue = \
                    policy_opt.equivocal_deception()
    else:
        print ("Undefined Deception Type")
        
    assert(deceptive_occupancy_measures >= 0).all()
    assert(deceptive_policy >= 0).all()


    ## 3. Solve IRL
    print("Solving IRL...")
    trajectories = irl.generate_trajectories(deceptive_policy, irl_num_traj, irl_len_traj)
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
        'observed_trajectory': trajectories,
        'transition_matrix': mmdp.transition_matrices
    }

    return result

if __name__ == '__main__':
    
    with mp.Pool() as pool:
        
        print([(beta_vec[i]) for i in range(len(beta_vec))])
        
        results = pool.starmap(run_all, [(beta_vec[i],) for i in range(len(beta_vec))])
        
        experiment_logger = {
            'beta_vec': beta_vec,
            'deception_type' : deception_type,
            # 'results' : results
        }
        
        save_file_name = 'result.pkl'
        save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name))
        with open(save_str, 'wb') as f:
            pickle.dump(experiment_logger, f, protocol=pickle.HIGHEST_PROTOCOL)
      

