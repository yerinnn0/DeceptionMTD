"""
Implements maximum entropy inverse reinforcement learning (Ziebart et al., 2008)
"""
from itertools import product
import time

import numpy as np
import numpy.random as rn


class IRL:
    
    def __init__(self, mmdp, epochs=5, learning_rate=0.9):
        self.mmdp = mmdp
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.feature_matrix = np.identity(mmdp.n_joint_states*mmdp.n_joint_actions) # (n_states * n_actions, n_states * n_actions)

    def generate_trajectories(self, policy, num_traj, len_traj, reset_s0 = True):
        """
        Generates trajectories based on given policy in MMDP.
        
        policy: Stochastic policy of MMDP. np.array(n_states, n_actions).
        num_traj: Number of trajectory to generate. int.
        len_traj: Length of trajectory to generate. int
        reset_s0: Whether to reset the initial state. bool.
        """

        trajectories = np.zeros([num_traj, len_traj, 2])

        for k in range(num_traj):
            if reset_s0:
                self.mmdp.reset_initial_state()
            trajectories[k, :, 0], trajectories[k, :, 1], reward_final= self.mmdp.run_MApolicy(policy, len_traj)

        self.trajectories = trajectories

        return trajectories
    
    def calculate_value_function(self, reward, policy, theta=1e-8):
        """
        Calculates value function with given reward and given policy
        
        reward: Reward matrix of MMDP. np.array with size(n_states, n_actions)
        policy: Stochastic policy of MMDP. np.array with size (n_states, n_actions)
        theta: Convergence threshold. float.
        
        return: Calculated value function. np.array with size (n_states,)
        """
        
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        gamma = self.mmdp.gamma
        transition_matrix = self.mmdp.transition_matrices
        
        reward = reward.reshape(n_states,n_actions)

        value_function = np.zeros(n_states)

        while True:
            delta = 0

            # Update the value function for each state
            for s in range(n_states):
                v = value_function[s]
                q_values = np.zeros(n_actions)
                
                # Compute Q(s,a) for all actions
                for a in range(n_actions):
                    # Compute expected value of next state
                    next_state_probs = transition_matrix[s][a] #(n_states)
                    expected_value = gamma * np.dot(next_state_probs, value_function)

                    # Add action-dependent reward to Q-value calculation
                    q_values[a] = reward[s,a] + expected_value
                
                q_values = q_values - np.max(q_values)
                
                value_function[s]= np.log(np.sum(np.exp(q_values)))
                
                policy[s] = np.exp(q_values)/np.exp(value_function[s])
                
                # Update delta
                delta = max(delta, abs(v - value_function[s]))

            # Check for convergence
            if delta < theta:
                break
            
        return value_function
    
    def estimate_value_function(self, trajectories):
        """
        Estimates value function using IRL based on given trajectories.
        
        trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
        
        return: Estimated value function. np.array with shape (n_states,)
        """
        
        n_states = self.mmdp.n_joint_states
        n_actions = self.mmdp.n_joint_actions
        discount = self.mmdp.gamma
        true_rewards = self.mmdp.joint_rewards
        transition_probability = self.mmdp.transition_matrices

        time0 = time.time()
        
        expected_rewards = irl(self.feature_matrix, n_actions, discount, transition_probability,
                    trajectories, self.epochs, self.learning_rate) # (n_states * n_actions, )
        # estimated_policy  = np.argmax(value_iteration(expected_rewards.reshape(n_states,-1), n_states, n_actions, transition_probability)[1],
        #                                axis = -1) # (n_states, ) # deterministic
        estimated_policy  = value_iteration(expected_rewards.reshape(n_states,-1), n_states, n_actions, transition_probability)[1] 
        estimated_value_function = self.calculate_value_function(true_rewards, estimated_policy)

        print("Time :", time.time()-time0)

        self.expected_rewards = expected_rewards
        self.estimated_policy = estimated_policy
        self.estimated_value_function = estimated_value_function
        self.self_estimated_value_function = self.calculate_value_function(self.expected_rewards, estimated_policy)
        
        return estimated_value_function


def value_iteration(reward, n_states, n_actions, transition_matrix, gamma=0.99, theta=1e-8):
        """
        Runs the value iteration algorithm to find the optimal value function and policy.

        reward: Reward. np.array with shape (n_states, n_actions).
        n_states: Number of states in MMDP. int.
        n_actions: Number of actions in MMDP. int.
        transition_matrix: Transition probability matrix of MMDP. np.array with shape (n_joint_states, n_joint_actions, n_joint_states).
        gamma: Discount factor of MMDP. float.
        theta: Convergence threshold. float.
        
        return: Tuple containing two np.arrays. The first array is of shape (n_states,) and represents the optimal
                value function. The second array is of shape (n_states, n_actions) and represents the optimal policy.
        """
        # Initialize the value function and policy arrays
        value_function = np.ones(n_states)
        policy = np.zeros((n_states, n_actions))
        
        while True:
            delta = 0

            # Update the value function for each state
            for s in range(n_states):
                v = value_function[s]
                q_values = np.zeros(n_actions)
                
                # Compute Q(s,a) for all actions
                for a in range(n_actions):
                    # Compute expected value of next state
                    next_state_probs = transition_matrix[s][a] #(n_states)
                    expected_value = gamma * np.dot(next_state_probs, value_function)

                    # Add action-dependent reward to Q-value calculation
                    q_values[a] = reward[s,a] + expected_value
                    
                q_values = q_values - np.max(q_values)
                
                value_function[s]= np.log(np.sum(np.exp(q_values)))
                
                policy[s] = np.exp(q_values)/np.exp(value_function[s])
                
                # Update delta
                delta = max(delta, abs(v - value_function[s]))

            # Check for convergence
            if delta < theta:
                break

        return value_function, policy
    
def irl(feature_matrix, n_actions, discount, transition_probability, trajectories, epochs, learning_rate):
    """
    Find the reward function for the given trajectories.

    feature_matrix: Matrix with the nth row representing the nth state. np.array with shape (n_states*n_action, 1)
    n_actions: Number of actions. int.
    discount: Discount factor of the MMDP. float.
    transition_probability: Transition probability matrix of MMDP. np.array with shape (n_joint_states, n_joint_actions, n_joint_states).
    trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
    epochs: Number of gradient descent steps. int.
    learning_rate: Gradient descent learning rate. float.
    return: Reward vector with shape (n_states * n_actions,).
    """

    n_states_actions, d_states_actions = feature_matrix.shape # n_states * n_actions
    n_states = int(feature_matrix.shape[0]/n_actions)

    # Initialise weights.
    alpha = rn.uniform(size=(d_states_actions,)) # Reward: (n_states * n_actions,)

    # Calculate the feature expectations \tilde{phi}.
    feature_expectations = find_feature_expectations(feature_matrix, trajectories, n_actions) # (n_states * n_actions, )
    
    # trajectories : (traj_number, steps, 2)

    # Gradient descent on alpha.
    for i in range(epochs):
        print("epoch:", i)
        r = feature_matrix.dot(alpha)
        expected_svf = find_expected_svf(n_states, r, n_actions, discount,
                                         transition_probability, trajectories)  # (n_states * n_actions, )
        grad = feature_expectations - feature_matrix.T.dot(expected_svf)   # (n_states * n_actions, )

        alpha += learning_rate * grad   # (n_states * n_actions, )

    return feature_matrix.dot(alpha).reshape((n_states * n_actions,))  # (n_states * n_actions, )

def find_svf(n_states, trajectories):
    """
    Find the state visitation frequency from trajectories.

    n_states: Number of states. int.
    trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
    
    return: State visitation frequencies vector with shape (n_states,).
    """

    svf = np.zeros(n_states)

    for trajectory in trajectories:
        for state, _, _ in trajectory:
            svf[state] += 1

    svf /= trajectories.shape[0]

    return svf

def find_feature_expectations(feature_matrix, trajectories, n_actions):
    """
    Find the feature expectations for the given trajectories. This is the average path feature vector.

    feature_matrix: Matrix with the nth row representing the nth state. np.array with shape (n_states*n_action, 1)
    trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
    
    return: Feature expectations vector with shape (n_states * n_actions,).
    """

    feature_expectations = np.zeros(feature_matrix.shape[1]) # (n_states * n_actions,)

    if len(trajectories.shape) == 4:
        trajectories = trajectories[0,:,:,:] # trajectories : (traj_number, steps, 2)

    for trajectory in trajectories: # (steps, 2)
        #for state, action in trajectory:
        for step in trajectory:
            state = step[0]
            action = step[1]
            feature_expectations += feature_matrix[int(state * n_actions + action)]

    feature_expectations /= trajectories.shape[0]

    return feature_expectations

def find_expected_svf(n_states, r, n_actions, discount,
                      transition_probability, trajectories):
    """
    Find the expected state visitation frequencies using algorithm 1 from
    Ziebart et al. 2008.

    n_states: Number of states. int.
    r: Reward. NumPy array with shape (n_states*n_actions,).
    n_actions: Number of actions. int.
    discount: Discount factor of the MMDP. float.
    transition_probability: Transition probability matrix of MMDP. np.array with shape (n_joint_states, n_joint_actions, n_joint_states).
    trajectories: 3D array of state/action pairs. States are ints, actions are ints. np.array with shape (num_traj, len_traj, 2).
    
    return: Expected state visitation frequencies vector with shape # (n_states * n_action, ).
    """
    
    if len(trajectories.shape) == 4:
        trajectories = trajectories[0,:,:,:]

    n_trajectories = trajectories.shape[0]
    trajectory_length = trajectories.shape[1]
    
    policy = value_iteration(r.reshape(n_states,-1), n_states, n_actions, transition_probability)[1]# (n_states,)
    
    start_state_action_count = np.zeros(n_states * n_actions) # Initial Distribution
    for trajectory in trajectories: # (step, 2)
        state = int(trajectory[0, 0])
        action = int(trajectory[0, 1])
        start_state_action_count[state*n_actions + action] += 1
    p_start_state_action = start_state_action_count/n_trajectories

    expected_svf = np.tile(p_start_state_action, (trajectory_length, 1)).T # (n_states * n_action, step, 1)
    
    for t in range(1, trajectory_length):
        expected_svf[:, t] = 0 # (n_states * n_action, 1)
        for i, j, k in product(range(n_states), range(n_actions), range(n_states)):
            expected_svf[i*n_actions + j, t] += (np.sum(expected_svf[k*n_actions:(k+1)*n_actions , t-1]) *
                                policy[i, j] * transition_probability[i, j, k]) * discount

    return expected_svf.sum(axis=1) # (n_states * n_action, )

def softmax(x1, x2):
    """
    Soft-maximum calculation, from algorithm 9.2 in Ziebart's PhD thesis.

    x1: float.
    x2: float.
    
    return: softmax(x1, x2)
    """
    max_x = max(x1, x2)
    min_x = min(x1, x2)
    return max_x + np.log(1 + np.exp(min_x - max_x))


