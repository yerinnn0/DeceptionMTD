import numpy as np
import time

# Parameters

## Probabilities
P_T = 0.1
P_A = 0.2
P_D = 0.6
P_B = 0.4
P_E = 0.2

## Rewards & Costs: Original Problem
R = 10
R_D = 5
C_T = 0.1
C_E = 3
C_B = 4
C_R = 20
C_D = 10 

## Rewards & Costs: Modified 
# R = 10
# R_D = 2
# C_T = 0.4
# C_E = 12
# C_B = 16
# C_R = 10
# C_D = 5

## Rewards & Costs for decoy
decoy_ratio = 0.01
R_decoy = R* decoy_ratio *10
R_D_decoy = R_D* decoy_ratio
C_T_decoy = C_T* decoy_ratio
C_E_decoy = C_E* decoy_ratio
C_B_decoy = C_B* decoy_ratio
C_R_decoy = C_R* decoy_ratio
C_D_decoy = C_D* decoy_ratio


def create_rewards():
    """
    Creates a reward matrix of size (n_states * n_actions) for a real agent.
    
    return: rewards. np.array with shape (n_states, n_actions)
    """
    # states = ["N", "T", "E", "B"]
    # actions = ["wait", "defend","reset"]
    n_states = 4
    actions = [0, 1, 2]

    # rewards: (n_states, n_actions)
    rewards = R*np.ones([n_states, len(actions)]) # baseline reward for single agent

    # Reward for wait action:
    rewards[:,0] = [R, R-C_T, R-C_E, R-C_B]
    # Reward for defend action:
    rewards[:,1] = [R+R_D-C_D, R+R_D-C_T-C_D, R+R_D-C_E-C_D, R+R_D-C_D-C_B]
    # Reward for reset action:
    rewards[:,2] = R - C_R

    return rewards

def create_decoy_rewards():
    """
    Creates a reward matrix of size (n_states * n_actions) for a decoy agent.
    
    return: rewards. np.array with shape (n_states, n_actions)
    """
    # states = ["N", "T", "E", "B"]
    # actions = ["wait", "defend","reset"]
    n_states = 4
    actions = [0, 1, 2]

    # rewards: (n_states, n_actions)
    rewards = R*np.ones([n_states, len(actions)]) # baseline reward for single agent
    
    # Reward for wait action:
    rewards[:,0] = [R_decoy, R_decoy-C_T_decoy, R_decoy-C_E_decoy, R_decoy-C_B_decoy]
    # Reward for defend action:
    rewards[:,1] = [R_decoy+R_D_decoy-C_D_decoy, R_decoy+R_D_decoy-C_T_decoy-C_D_decoy, R_decoy+R_D_decoy-C_E_decoy-C_D_decoy, R_decoy+R_D_decoy-C_D_decoy-C_B_decoy]
    # Reward for reset action:
    rewards[:,2] = R_decoy - C_R_decoy

    return rewards

def select_value_from_distribution(N_agents, n_states, distribution):
    """
    Selects a value from the given distribution.
    """

    states = []
    for j in range(N_agents):
        distribution_for_agent = distribution[n_states*j:n_states*(j+1)]
        r = np.random.uniform(0,1)
        for i in range(len(distribution_for_agent)):
            if r < np.sum(distribution_for_agent[:i+1]):
                states.append(i)
                break

    return states

class MultiAgentGridworld:
    # perturbation = 0 gives input perturbation, and perturbation = 1 gives output perturbation
    def __init__(self, N_agents, initial_distribution, n_states, n_actions, rewards, gamma, v_reach = 0.9, variance = 0, p=0.1, perturbation=0):
        self.N_agents = N_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.s0 = self.get_joint_state(select_value_from_distribution(N_agents, n_states, initial_distribution))
        self.rewards = rewards
        self.variance = variance
        self.p = p
        self.gamma = gamma
        self.perturbation = perturbation
        self.v_reach = v_reach
        
        self.goal_states = None
        self.decoy_states = None
        
        self.n_joint_states = self.n_states**self.N_agents
        self.n_joint_actions = self.n_actions**self.N_agents

        self.initial_distribution = np.zeros(self.n_joint_states)
        self.initial_distribution[self.s0] = 1
        
        self.build_joint_transition_matrix()
        self.build_joint_rewards()

    def reset_initial_state(self):
        self.s0 = select_value_from_distribution(1, self.n_states, self.initial_distribution)[0]
        
    def set_goal_states(self, goal_states, decoy_states):
        self.goal_states = goal_states
        self.decoy_states = decoy_states

    def joint_state_space(self):
        return np.arange(self.n_joint_states)

    def agent_state_space(self):
        return np.arange(self.n_states)

    def agent_action_space(self):
        return np.arange(self.n_actions)

    def joint_action_space(self):
        return np.arange(self.n_joint_actions)

    def get_agent_slice(self, agent_id):
        return slice(agent_id*self.n_states, (agent_id+1)*self.n_states)

    def get_joint_state(self, states):
        # returns a joint state given a list of local states
        return np.ravel_multi_index(states, [self.n_states]*self.N_agents)

    def get_joint_action(self, actions):
        # returns a joint action given a list of local actions
        return np.ravel_multi_index(actions, [self.n_actions]*self.N_agents)

    def get_state(self, joint_state):
        # returns a list of local states for each agent
        return [np.unravel_index(joint_state, [self.n_states]*self.N_agents)[agent_id] for agent_id in range(self.N_agents)]

    def get_action(self, joint_action):
        # returns a list of local actions for each agent
        return [np.unravel_index(joint_action, [self.n_actions]*self.N_agents)[agent_id] for agent_id in range(self.N_agents)]

    def get_next_state(self, joint_state, joint_action):
        # Obtain probability distribution of next states
        next_state_probs = self.joint_transition_matrix[joint_state,
                                                        joint_action, :]
        # Sample from probability distribution
        next_state = np.random.choice(self.n_joint_states, p=next_state_probs)
        return next_state

    def build_agent_transition_matrix(self, p=0.1):
        transition_matrix = np.zeros(
            (self.n_states, self.n_actions, self.n_states))

        # Fill in transition matrix

        transition_matrix[:,0,:] = [[1-P_T, P_T, 0, 0],[0, 1-P_E, P_E, 0], [0, 0, 1-P_B, P_B],[0, 0, 0, 1]]
        transition_matrix[:,1,:] = [[1-P_T, P_T, 0, 0],[P_D, (1-P_D)*(1-P_E), (1-P_D)*P_E, 0], [P_D, 0, (1-P_D)*(1-P_B), (1-P_D)*P_B],[0, 0, 0, 1]]
        transition_matrix[:,2,:] = [[1, 0, 0, 0],[1, 0, 0, 0], [1, 0, 0, 0],[1, 0, 0, 0]]

        self.agent_transition_matrix = transition_matrix

        return self.agent_transition_matrix

    def build_joint_transition_matrix(self):
        self.transition_matrices = []
        # Initialize joint transition matrix to all ones
        self.joint_transition_matrix = np.ones(
            (self.n_joint_states, self.n_joint_actions, self.n_joint_states))
        for i in range(self.N_agents):
            # Build a transition matrix for each agent
            self.transition_matrices.append(
                self.build_agent_transition_matrix())
        time0 = time.time()
        for s in range(self.n_joint_states):
            local_current_state = self.get_state(s)
            for a in range(self.n_joint_actions):
                local_action = self.get_action(a)
                for y in range(self.n_joint_states):
                    local_next_state = self.get_state(y)
                    # Get the local state of each agent
                    for agent_id in range(self.N_agents):
                        # Make each element of the joint transition matrix the product of the local transition matrices
                        self.joint_transition_matrix[s, a, y] *= self.transition_matrices[agent_id][local_current_state[agent_id],
                                                                                                    local_action[agent_id], local_next_state[agent_id]]
                        
        self.transition_matrices = np.array(self.joint_transition_matrix).squeeze() #(n_states, n_actions, n_states)

        print("Time to build joint transition matrix:", time.time()-time0)

    def build_joint_rewards(self):
        # Initialize joint transition matrix to all ones
        self.joint_rewards = np.zeros(
            (self.n_joint_states, self.n_joint_actions))
        time0 = time.time()
        for s in range(self.n_joint_states):
            local_current_state = self.get_state(s)
            for a in range(self.n_joint_actions):
                local_action = self.get_action(a)
                # Get the local state of each agent
                for agent_id in range(self.N_agents):
                    # Make each element of the joint transition matrix the product of the local transition matrices
                    self.joint_rewards[s, a] += self.rewards[:,:,agent_id][local_current_state[agent_id],local_action[agent_id]]
                        
        self.joint_rewards = np.array(self.joint_rewards).squeeze() #(n_states, n_actions, n_states)

        print("Time to build joint joint rewards:", time.time()-time0)
                    

    def run_MApolicy(self, policy, n_steps):
        """
        Runs an input policy for N timesteps and outputs the state trajectory and reward history.

        policy: np.array of shape (n_states, n_actions) representing the policy. this is a stochastic policy.
        n_steps: int representing the number of timesteps to run the policy.
        
        return: Tuple containing two np.arrays. The first array is of shape (n_steps,) and represents the state 
                 trajectory. The second array is of shape (n_steps,) and represents the reward history.
        """
        state = self.s0
        state_traj = np.zeros(n_steps, dtype=int)
        action_traj = np.zeros(n_steps, dtype=int)
        reward_traj = np.zeros(n_steps)

        for i in range(n_steps):
            action_probs = policy[state, :]/np.sum(policy[state, :])  # gives idx of joint action
            # action_probs = np.zeros(policy[state,:].shape)
            # action_probs[np.argmax(policy[state,:])] = 1

            # selects an action based on the probability distribution
            action = np.random.choice(self.n_joint_actions, p=action_probs)

            # Apply action and get next state and reward
            # gets next state based on initial state and chosen action
            next_state = self.get_next_state(state, action)
            reward = self.joint_rewards[state, action]
            # reward = self.get_reward(state, action)

            # Update state trajectory and reward history
            state_traj[i] = state
            reward_traj[i] = reward
            action_traj[i] = action

            # Update current state
            state = next_state
        reward_final = np.sum(reward_traj)
        return state_traj, action_traj, reward_final

