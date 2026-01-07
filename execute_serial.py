
import numpy as np
import multiprocessing as mp
import json
import pickle
import os
import gc
import sys
import torch

import time

from config import *

from MDP import create_rewards, create_decoy_rewards, MultiAgentGridworld
# from cp_MDP import create_rewards, create_decoy_rewards, MultiAgentGridworld

from policy_optimization.policy_optimization_scipy import PolicyOptimizationScipy
from policy_optimization.policy_optimization_cvxopt import PolicyOptimizationCVXOPT
from policy_optimization.policy_optimization_osqp import PolicyOptimizationOSQP
from policy_optimization.policy_optimization_gd import PolicyOptimizationGD
from policy_optimization.policy_optimization_qpth import PolicyOptimizationQPTH
# from policy_optimization.policy_optimization_ipopt import PolicyOptimizationIPOPT
from policy_optimization.policy_optimization_gurobi import PolicyOptimizationGurobi
from policy_optimization.policy_optimization_pyomo import PolicyOptimizationPyomo

from irl.maxent_irl import MaxEntIRL
# from irl.linear_irl import LinearIRL
from irl.deep_maxent_irl import DeepMaxEntIRL
from irl.apprenticeship_learning import ApprenticeshipIRL

from osqp import algebras_available

def run_all_serial(beta_vec):

    def move_dict_to_cpu(d):
        """
        Recursively move all torch tensors in a nested dictionary to CPU.
        """
        if isinstance(d, dict):
            return {k: move_dict_to_cpu(v) for k, v in d.items()}
        elif isinstance(d, torch.Tensor):
            return d.detach().cpu()  # detach() if it requires grad
        else:
            return d

    def make_json_serializable(obj):

        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            # Avoid recursion for flat lists of simple types
            if all(isinstance(v, (int, float, str, bool, type(None))) for v in obj):
                return obj
            return [make_json_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()  # Convert numpy scalar to Python scalar
        elif hasattr(obj, 'tolist'):
            return obj.tolist()
        else:
            return obj
    
    def make_json_serializable_old(obj):
        # Convert non-serializable objects here (e.g., NumPy arrays, custom objects)
        if isinstance(obj, dict):
            return {k: make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_json_serializable(v) for v in obj]
        elif hasattr(obj, 'tolist'):  # e.g., NumPy arrays
            return obj.tolist()
        else:
            return obj
            
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
            agent_states = []
            for k in range(n_local_states):
                agent_states.append(find_the_joint_state(n_local_states, N_agents, i, k))
            goal_states.append(agent_states)

        for i in target_decoy_agents:
            agent_states = []
            for k in range(n_local_states):
                agent_states.append(find_the_joint_state(n_local_states, N_agents, i, k))
            decoy_states.append(agent_states)

        # print("Goal states:", np.array(goal_states).shape, "Decoy states:", np.array(decoy_states).shape)
        
        return np.array(goal_states), np.array(decoy_states) # (n_real_agents, n_local_states, n_joint_states), (n_decoy_agents, n_local_states, n_joint_states)
    
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
    
    
    time0 = time.time()

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
    
    mmdp = MultiAgentGridworld(N_agents, initial_distribution, n_local_states, n_local_actions, rewards, gamma, v_reach, build_transition_matrix=build_transition_matrix)
    
    mmdp.set_goal_states(goal_states, decoy_states)
    
    print("Time for setting up MDP :", time.time()-time0)
    
    # 1. Solve LP
    lp_file_name = save_file_format['lp']
    lp_file_name_json = save_file_format['lp'][:-4] +".json"


    if lp_file_name in os.listdir():

        print("Reading "+ lp_file_name)

        time0 = time.time()

        save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), lp_file_name))
        with open(save_str, 'rb') as f:
            exp_logger = pickle.load(f)
            
        lp_result = exp_logger['results']

        occupancy_measures = lp_result[0]["occupancy_measure"]["occupancy_measure"]
        policy = lp_result[0]["policy"]["policy"]
        value_function = lp_result[0]["value_function"]["value_function"]
        revenue = lp_result[0]["revenue"]["revenue"]


        if not build_transition_matrix:
            mmdp.joint_transition_matrix = np.array(lp_result[0]['transition_matrix'])
            mmdp.transition_matrices = np.array(mmdp.joint_transition_matrix).squeeze()

        print("MDP results loaded, Time : ", time.time()-time0)
    
    elif lp_file_name_json in os.listdir():
        time0 = time.time()

        save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), lp_file_name_json))
        with open(save_str, 'r') as f:
            exp_logger = json.load(f)
            
        lp_result = exp_logger['results']
        occupancy_measures = np.array(lp_result[0]["occupancy_measure"]["occupancy_measure"])
        policy = np.array(lp_result[0]["policy"]["policy"])
        value_function = np.array(lp_result[0]["value_function"]["value_function"])
        revenue = lp_result[0]["revenue"]["revenue"]

        if not build_transition_matrix:
            mmdp.joint_transition_matrix = np.array(lp_result[0]['transition_matrix'])
            mmdp.transition_matrices = np.array(mmdp.joint_transition_matrix).squeeze()

        print("MDP results loaded, Time : ", time.time()-time0)
        
    else:

        print("LP result not found")

        if optimization_solver['mdp'] == "scipy":
            mdp_solver = PolicyOptimizationScipy(mmdp)
        elif optimization_solver['mdp'] == "cvxopt":
            mdp_solver = PolicyOptimizationCVXOPT(mmdp)
        elif optimization_solver['mdp'] == "osqp":
            mdp_solver = PolicyOptimizationOSQP(mmdp)
        elif optimization_solver['mdp'] == "gd":
            mdp_solver = PolicyOptimizationGD(mmdp)
        if optimization_solver['mdp'] == "gurobi":
            mdp_solver = PolicyOptimizationGurobi(mmdp)
        if optimization_solver['mdp'] == "pyomo":
            mdp_solver = PolicyOptimizationPyomo(mmdp)

            
        occupancy_measures, policy, value_function, revenue =mdp_solver.solve_MDP()
        
        if SAVE_INTERMEDIATE_RESULT:

            print("Saving result")
        
            result = {
                'config' : {
                    'MMDP_SETTINGS' : MMDP_SETTINGS,
                    'IRL_SETTINGS' : IRL_SETTINGS,
                    "DECEPTION_SETTINGS" : DECEPTION_SETTINGS,
                    "OPTIMIZATION_SETTINGS" : OPTIMIZATION_SETTINGS
                },
                'value_function':{
                    'value_function': value_function
                },
                'policy':{
                    'policy': policy
                },
                'revenue':{
                    'revenue': revenue
                },
                'reward':{
                    'reward': mmdp.joint_rewards
                },
                'occupancy_measure':{
                    'occupancy_measure':occupancy_measures
                },
                'transition_matrix': mmdp.transition_matrices
            }

            return [move_dict_to_cpu(result)]


    ## 2. Solve QP
    
    deception_file_name = save_file_format['deception']
    deception_file_name_json = save_file_format['deception'][:-4] +".json"
    
    if deception_file_name in os.listdir():
        pass

    elif deception_file_name_json in os.listdir():
        pass

    else:
        print("Deception result not found")

        if optimization_solver[deception_type] == "scipy":
            deception_solver = PolicyOptimizationScipy(mmdp)
        elif optimization_solver[deception_type] == "cvxopt":
            deception_solver = PolicyOptimizationCVXOPT(mmdp)
        elif optimization_solver[deception_type] == "osqp":
            deception_solver = PolicyOptimizationOSQP(mmdp)
        # elif optimization_solver[deception_type] == "ipopt":
        #     deception_solver = PolicyOptimizationIPOPT(mmdp)
        elif optimization_solver[deception_type] == "gd":
            deception_solver = PolicyOptimizationGD(mmdp)
        elif optimization_solver[deception_type] == "qpth":
            deception_solver = PolicyOptimizationQPTH(mmdp)
        if optimization_solver[deception_type] == "gurobi":
            deception_solver = PolicyOptimizationGurobi(mmdp)
        if optimization_solver[deception_type] == "pyomo":
            deception_solver = PolicyOptimizationPyomo(mmdp)

        # deception_solver = None
            
        results = []
        for beta in beta_vec:

            print(beta)

            if deception_type == "diversionary":
                deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue = \
                            deception_solver.diversionary_deception(occupancy_measures, init = occupancy_measures, beta = beta)
            elif deception_type == "targeted":
                deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue = \
                            deception_solver.targeted_deception(target_occupancy_measure, original_occupancy_measures = occupancy_measures, beta = beta)
            elif deception_type == "equivocal":
                deceptive_occupancy_measures, deceptive_policy, deceptive_value_function, deceptive_revenue = \
                            deception_solver.equivocal_deception(original_occupancy_measures = occupancy_measures, beta = beta)
            else:
                print ("Undefined Deception Type")
            
            assert(deceptive_occupancy_measures >= 0).all()
            assert(deceptive_policy >= 0).all()

            if SAVE_INTERMEDIATE_RESULT:
            
                result = {
                    'config' : {
                        'MMDP_SETTINGS' : MMDP_SETTINGS,
                        'IRL_SETTINGS' : IRL_SETTINGS,
                        "DECEPTION_SETTINGS" : DECEPTION_SETTINGS,
                        "OPTIMIZATION_SETTINGS" : OPTIMIZATION_SETTINGS
                    },
                    # 'value_function':{
                    #     'value_function': value_function,
                    #     'deceptive_value_function': deceptive_value_function
                    # },
                    'policy':{
                        'policy': policy,
                        'deceptive_policy': deceptive_policy
                    },
                    'revenue':{
                        'revenue': revenue,
                        'deceptive_revenue':deceptive_revenue
                    },
                    'reward':{
                        'reward': mmdp.joint_rewards
                    },
                    'occupancy_measure':{
                        'occupancy_measure':occupancy_measures,
                        'deceptive_occupancy_measure': deceptive_occupancy_measures,
                        'target_occupancy_measure': target_occupancy_measure
                    },
                    'transition_matrix': mmdp.transition_matrices
                }
                
                time0 = time.time()
                results.append(move_dict_to_cpu(result))
                # results.append(make_json_serializable(result))
                # print("Time for making json serializable:", time.time()-time0)

                # Save
                simulation_type = 'deception'
                save_file_name = save_file_format[simulation_type][:-4]+"_"+str(np.round(beta,2))+".pkl"
                save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name))
                with open(save_str, 'wb') as f:
                    pickle.dump(results[-1], f, protocol=pickle.HIGHEST_PROTOCOL)
                print(save_file_name + " Saved")

        if SAVE_INTERMEDIATE_RESULT:
            
            return results

    ## 3. Solve IRL
    print("Setting IRL")
    if irl_model == "maxent":
        irl = MaxEntIRL(mmdp, feature_map= feature_map, epochs = irl_epoch, learning_rate = irl_learning_rate)
    # elif irl_model == "linear":
    #     irl = LinearIRL(mmdp)
    elif irl_model == "deep_maxent":
        irl = DeepMaxEntIRL(mmdp, feature_map= feature_map,
                            epochs = irl_epoch, learning_rate = irl_learning_rate, layers = irl_layers)
    elif irl_model == "apprenticeship":
        irl = ApprenticeshipIRL(mmdp, feature_map= feature_map, threshold = 1, max_iter = irl_max_iter)
        
    print("Reading deception result...")

    deception_file_name = save_file_format['deception']
    deception_file_name_json = save_file_format['deception'][:-4] +".json"
    if deception_file_name in os.listdir():
        save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), deception_file_name))
        with open(save_str, 'rb') as f:
            exp_logger = pickle.load(f)
    elif deception_file_name_json in os.listdir():
        save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), deception_file_name_json))
        with open(save_str, 'r') as f:
            exp_logger = json.load(f)

    beta_vec = exp_logger['beta_vec']
    deception_result = exp_logger['results']
        
    print("Solving IRL...")

    results = []
    for i in range(len(beta_vec)):

        print(i)

        deceptive_occupancy_measures = deception_result[i]["occupancy_measure"]["deceptive_occupancy_measure"]
        deceptive_policy = deception_result[i]["policy"]["deceptive_policy"]
        # deceptive_value_function = deception_result[i]["value_function"]["deceptive_value_function"]
        deceptive_revenue = deception_result[i]["revenue"]["deceptive_revenue"]

        if isinstance(deceptive_occupancy_measures, torch.Tensor):
            deceptive_occupancy_measures = deceptive_occupancy_measures.cpu().numpy()
        else:
            deceptive_occupancy_measures = np.array(deceptive_occupancy_measures)
        if isinstance(deceptive_policy, torch.Tensor):
            deceptive_policy = deceptive_policy.cpu().numpy()
        else:
            deceptive_policy = np.array(deceptive_policy)
        # if isinstance(deceptive_value_function, torch.Tensor):
        #     deceptive_value_function = deceptive_value_function.cpu().numpy()
        # else:
        #     deceptive_value_function = np.array(deceptive_value_function)
        if isinstance(deceptive_revenue, torch.Tensor):
            deceptive_revenue = deceptive_revenue.cpu().numpy()
        else:
            deceptive_revenue = np.array(deceptive_revenue)

        # print("Generating Trajectories")
        trajectories = irl.generate_trajectories_batched(deceptive_policy, irl_num_traj, irl_len_traj)
        # print("Generated Trajectories")
        estimated_value_function = irl.estimate_value_function(trajectories.unsqueeze(0))
        # estimated_value_function = irl.estimate_value_function(np.expand_dims(trajectories, axis = 0))
  
            
        # scale = np.sum(np.abs(value_function))
        # deception_error = calculate_error(value_function, deceptive_value_function, scale)
        # inverse_learning_error = calculate_error(deceptive_value_function, estimated_value_function, scale)
        # total_estimation_error = calculate_error(value_function, estimated_value_function, scale)
        
        # policy_difference = np.sum(np.abs(policy - deceptive_policy))
        
        result = {
            'config' : {
                    'MMDP_SETTINGS' : MMDP_SETTINGS,
                    'IRL_SETTINGS' : IRL_SETTINGS,
                    "DECEPTION_SETTINGS" : DECEPTION_SETTINGS,
                    "OPTIMIZATION_SETTINGS" : OPTIMIZATION_SETTINGS
                },
            # 'error' : {
            #     'deception_error': deception_error, 
            #     'inverse_learning_error': inverse_learning_error, 
            #     'total_estimation_error': total_estimation_error
            # },
            # 'value_function':{
            #     'value_function': value_function,
            #     'deceptive_value_function': deceptive_value_function,
            #     'estimated_value_function': estimated_value_function,
            #     # 'self_estimated_value_function': irl.self_estimated_value_function
            # },
            'policy':{
                # 'policy': policy,
                # 'deceptive_policy': deceptive_policy,
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
            # 'occupancy_measure':{
            #     'occupancy_measure':occupancy_measures,
            #     'deceptive_occupancy_measure': deceptive_occupancy_measures,
            #     'target_occupancy_measure': target_occupancy_measure,
            # },
            # 'policy_difference': np.array(policy_difference),
            # 'observed_trajectory': trajectories,
            # 'transition_matrix': mmdp.transition_matrices
        }

        results.append(move_dict_to_cpu(result))
        # results.append(make_json_serializable(result))

    return results

if __name__ == '__main__':

    print("Python executable:", sys.executable)
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    # os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
    
    # print(torch.cuda.is_available())

    # Assume LP and DECEPTION results exists
    for k in irl_repetitions:
        results= run_all_serial(beta_vec)
            
        experiment_logger = {
            'beta_vec': beta_vec.tolist(),
            'deception_type' : str(deception_type),
            'results' : results
        }
            
        if 'estimated_reward' in results[0]['reward']:
            simulation_type = 'irl'
        elif 'deceptive_policy' in results[0]['policy']:
            simulation_type = 'deception'
            save_file_name = save_file_format[simulation_type]
            save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name))
            with open(save_str, 'wb') as f:
                pickle.dump(experiment_logger, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(save_file_name + " Saved")
            break
        else:
            simulation_type = 'lp'
            save_file_name = save_file_format[simulation_type]
            save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name))
            with open(save_str, 'wb') as f:
                pickle.dump(experiment_logger, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(save_file_name + " Saved")
            break
                
        save_file_name = save_file_format[simulation_type][:-4] + "_"+str(k)+".pkl" 
        save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name))
        with open(save_str, 'wb') as f:
            pickle.dump(experiment_logger, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(save_file_name + " Saved")

    # save_file_name_json = save_file_format[simulation_type][:-4] +".json"
    # save_str_json = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name_json))
    # with open(save_str_json, 'w') as f: 
    #     json.dump(experiment_logger, f, indent=2)
    # print(save_file_name_json + " Saved")
        
        
    while simulation_type != 'irl':

        for k in irl_repetitions:
            results= run_all_serial(beta_vec)
                
            experiment_logger = {
                'beta_vec': beta_vec.tolist(),
                'deception_type' : str(deception_type),
                'results' : results
            }
                
            if 'estimated_reward' in results[0]['reward']:
                simulation_type = 'irl'
            elif 'deceptive_policy' in results[0]['policy']:
                simulation_type = 'deception'
                save_file_name = save_file_format[simulation_type]
                save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name))
                with open(save_str, 'wb') as f:
                    pickle.dump(experiment_logger, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(save_file_name + " Saved")
                break
            else:
                simulation_type = 'lp'
                save_file_name = save_file_format[simulation_type]
                save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name))
                with open(save_str, 'wb') as f:
                    pickle.dump(experiment_logger, f, protocol=pickle.HIGHEST_PROTOCOL)
                print(save_file_name + " Saved")
                break
                    
            save_file_name = save_file_format[simulation_type][:-4] + "_"+str(k)+".pkl" 
            save_str = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name))
            with open(save_str, 'wb') as f:
                pickle.dump(experiment_logger, f, protocol=pickle.HIGHEST_PROTOCOL)
            print(save_file_name + " Saved")

        
        # save_file_name_json = save_file_format[simulation_type][:-4] +".json"
        # save_str_json = os.path.abspath(os.path.join(os.path.abspath(os.path.curdir), save_file_name_json))
        # with open(save_str_json, 'w') as f: 
        #     json.dump(experiment_logger, f, indent=2)
        # print(save_file_name_json + " Saved")
      

