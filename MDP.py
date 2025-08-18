import numpy as np
import time
import torch

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
decoy_ratio = 0.1
R_decoy = R* decoy_ratio
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

def select_value_from_distribution(N_agents, n_states, distribution, device):
    """
    Selects a value from the given distribution.
    """
    # # Ensure distribution is a tensor on device
    # if not isinstance(distribution, torch.Tensor):
    #     dist_t = torch.as_tensor(distribution, dtype=torch.float32, device=device)
    # else:
    #     dist_t = distribution.to(device).float()

    # # Reshape into (N_agents, n_states)
    dist_reshaped = distribution.view(N_agents, n_states)

    # # Sample 1 index per agent according to their distribution row using multinomial
    # multinomial expects probs sum to 1 (assumed here)
    samples = torch.multinomial(dist_reshaped, num_samples=1).squeeze(1)

    # Convert tensor to Python list
    return samples.cpu().tolist()

    # states = []
    # for j in range(N_agents):
    #     distribution_for_agent = distribution[n_states*j:n_states*(j+1)]
    #     r = np.random.uniform(0,1)
    #     for i in range(len(distribution_for_agent)):
    #         if r < np.sum(distribution_for_agent[:i+1]):
    #             states.append(i)
    #             break

    # return states

class MultiAgentGridworld:
    # perturbation = 0 gives input perturbation, and perturbation = 1 gives output perturbation
    def __init__(self, N_agents, initial_distribution, n_states, n_actions, rewards, gamma, v_reach = 0.9, variance = 0, p=0.1, perturbation=0, build_transition_matrix = True):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        self.initial_distribution = initial_distribution
        self.initial_distribution_t = torch.as_tensor(self.initial_distribution, dtype=torch.float32, device=self.device)
        
        self.N_agents = N_agents
        self.n_states = n_states
        self.n_actions = n_actions
        self.s0 = self.get_joint_state(select_value_from_distribution(N_agents, n_states, self.initial_distribution_t, self.device))
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

        self.initial_joint_distribution = np.zeros(self.n_joint_states)
        self.initial_joint_distribution[self.s0] = 1
        
        if build_transition_matrix:
            self.build_joint_transition_matrix()
        self.build_joint_rewards()
        
        self.joint_rewards_t = torch.as_tensor(self.joint_rewards, dtype=torch.float32, device=self.device)
        self.initial_joint_distribution_t = torch.as_tensor(self.initial_joint_distribution, dtype=torch.float32, device=self.device)
        self.joint_transition_matrix_t = torch.as_tensor(self.joint_transition_matrix, dtype=torch.float32, device=self.device)

    def feature_vector(self, i, feature_map="identity"):
        """
        Get the feature vector associated with a state integer.

        i: State int.
        feature_map: Which feature map to use (default identity). String in {identity,
            coord, proxi}.
        -> Feature vector.
        """
        grid_size = self.n_actions*self.n_states
        if feature_map == "agent_controlled":
            n_local_features = 2
            f = np.zeros(self.N_agents*n_local_features) 
            joint_state = i // self.n_joint_actions
            local_states = self.get_state(joint_state)
            for agent in range(self.N_agents):
                if local_states[agent] in [0,1]: # Not controlled by adversary
                    f[agent*n_local_features] = 1
                elif local_states[agent] in [2,3]: # Controlled by adversary
                    f[agent*n_local_features+1] = 1
            return f
            
        elif feature_map == "set_local":
            # f = np.zeros(self.n_states)
            f = np.zeros(grid_size)
            x, y = i % grid_size, i // grid_size
            # joint_state = i // self.n_joint_actions
            # local_states = self.get_state(joint_state)
            # joint_action = i % self.n_joint_actions
            # local_actions = self.get_action(joint_action)
            # print(y,local_actions[0]+self.n_actions*local_states[0])
            # print(x,local_actions[1]+self.n_actions*local_states[1])
            # for agent in range(self.N_agents):
            #     s = local_states[agent]
            #     a = local_actions[agent]
            #     f[s*self.n_actions + a] += 1
            s_list = [np.unravel_index(i, [self.n_states*self.n_actions]*self.N_agents)[agent_id] for agent_id in range(self.N_agents)]
            for s in s_list:
                f[s] += 1
            return f
        if feature_map == "coord":
            f = np.zeros(grid_size)
            x, y = i % grid_size, i // grid_size
            f[x] += 1
            f[y] += 1
            return f
        if feature_map == "proxi":
            f = np.zeros(self.n_joint_states*self.n_joint_actions) 
            x, y = i % grid_size, i // grid_size
            for b in range(grid_size):
                for a in range(grid_size):
                    dist = abs(x - a) + abs(y - b)
                    f[a+b*grid_size] = 1/(dist+1)
            return f
        if feature_map == "correlated":
            f = np.zeros((self.n_states,self.n_states))
            # joint_state = i // self.n_joint_actions
            # local_states = self.get_state(joint_state)
            j = i // self.n_joint_actions
            x, y = j % self.n_states, j // self.n_states
            for b in range(self.n_states):
                f[x,b] += 1
                f[b,y] += 1
            f = f.flatten()
            return f
        if feature_map == "local_state":
            f = np.zeros(self.N_agents*self.n_states) 
            for agent in range(self.N_agents):
                if agent == 0:
                    local_state = (i // self.n_joint_actions) // self.n_states
                if agent == 1:
                    local_state = (i // self.n_joint_actions) % self.n_states
                f[agent*self.n_states+local_state] = 1
            return f
        # Assume identity map.
        f = np.zeros(self.n_joint_states*self.n_joint_actions)
        f[i] = 1
        return f

    def feature_matrix(self, feature_map="identity"):
        """
        Get the feature matrix for this gridworld.

        feature_map: Which feature map to use (default identity). String in {identity,
            coord, proxi}.
        -> NumPy array with shape (n_states, d_states).
        """

        features = []
        for n in range(self.n_joint_states*self.n_joint_actions):
            f = self.feature_vector(n, feature_map)
            features.append(f)
        return np.array(features)

    def reset_initial_state(self):
        self.s0 = self.get_joint_state(select_value_from_distribution(self.N_agents, self.n_states, 
                                                                      self.initial_distribution_t, device = self.device))
        
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

    def get_next_state_cpu(self, joint_state, joint_action):
        # Obtain probability distribution of next states
        next_state_probs = self.joint_transition_matrix[joint_state,
                                                        joint_action, :]
        # Sample from probability distribution
        next_state = np.random.choice(self.n_joint_states, p=next_state_probs)
        return next_state
    
    def get_next_state(self, joint_state, joint_action):
        """
        GPU version of get_next_state.
        Assumes self.joint_transition_matrix is a torch.Tensor on device
        with shape (n_joint_states, n_joint_actions, n_joint_states).
        """

        probs = self.joint_transition_matrix_t.index_select(0, joint_state)  # (1, n_actions, n_states)
        probs = probs.index_select(1, joint_action)  # (1, 1, n_states)
        probs = probs.squeeze(0).squeeze(0)
        next_state = torch.multinomial(probs, num_samples=1).squeeze()  # tensor scalar on device

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

    def build_joint_transition_matrix_original(self):
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

    def build_joint_transition_matrix_cpu(self):
        # Get individual transition matrices (N_agents, n_states, n_actions, n_states)
        # Precompute maps:
        state_map = np.array([self.get_state(s) for s in range(self.n_joint_states)])  # (S, N_agents)
        action_map = np.array([self.get_action(a) for a in range(self.n_joint_actions)])  # (A, N_agents)

        # transition_matrices shape: (N_agents, n_states, n_actions, n_states)
        transition_matrices = np.array([
            self.build_agent_transition_matrix() for _ in range(self.N_agents)
        ])  # (N_agents, S_agent, A_agent, S_agent)

        joint_transitions_per_agent = []
        
        time0 = time.time()

        for agent_id in range(self.N_agents):
            # Index for agent:
            # For each joint state s, action a, next state y,
            # get agent's local current, action, next state
            s_idx = state_map[:, agent_id][:, None, None]   # shape (S,1,1)
            a_idx = action_map[:, agent_id][None, :, None]  # shape (1,A,1)
            y_idx = state_map[:, agent_id][None, None, :]   # shape (1,1,S)

            # Use advanced indexing to get transition prob for each triple (s,a,y)
            agent_trans = transition_matrices[agent_id, s_idx, a_idx, y_idx]  # shape (S,A,S)

            joint_transitions_per_agent.append(agent_trans)

        # Now compute elementwise product across agents (axis=0)
        joint_transition_matrix = np.prod(np.array(joint_transitions_per_agent), axis=0)  # (S,A,S)

        # self.build_joint_transition_matrix_original()
        # print(self.joint_transition_matrix.shape)
        # print(joint_trans.shape)

        self.joint_transition_matrix = joint_transition_matrix
        self.transition_matrices = np.array(self.joint_transition_matrix).squeeze()

        print("Time to build joint transition matrix:", time.time()-time0)

    def build_joint_transition_matrix(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Precompute local states and actions for all joint states and actions
        state_map = torch.tensor(
            np.array([self.get_state(s) for s in range(self.n_joint_states)]), 
            dtype=torch.long, device=device
        )  # shape (S, N_agents)

        action_map = torch.tensor(
            np.array([self.get_action(a) for a in range(self.n_joint_actions)]), 
            dtype=torch.long, device=device
        )  # shape (A, N_agents)

        # Build transition matrices for each agent and move to GPU
        transition_matrices = torch.stack([
            torch.tensor(self.build_agent_transition_matrix(), dtype=torch.float32, device=device)
            for _ in range(self.N_agents)
        ])  # shape (N_agents, S_agent, A_agent, S_agent)

        S, A = self.n_joint_states, self.n_joint_actions
        N = self.N_agents

        time0 = time.time()

        # joint_probs = torch.ones((S, A, S), device=device)

        # for agent_id in range(N):
        #     s_local = state_map[:, agent_id]   # (S,)
        #     a_local = action_map[:, agent_id]  # (A,)
        #     y_local = state_map[:, agent_id]   # (S,)

        #     # Expand dims to broadcast indexing: 
        #     # s_local: (S,1,1), a_local: (1,A,1), y_local: (1,1,S)
        #     s_exp = s_local[:, None, None]
        #     a_exp = a_local[None, :, None]
        #     y_exp = y_local[None, None, :]

        #     # Index into agent's transition matrix (S_agent, A_agent, S_agent)
        #     agent_trans = transition_matrices[agent_id][s_exp, a_exp, y_exp]  # (S, A, S)

        #     joint_probs *= agent_trans

        # self.joint_transition_matrix = joint_probs.cpu().numpy().squeeze()
        # self.transition_matrices = np.array(self.joint_transition_matrix).squeeze()

        chunk_size = 500
        joint_probs = torch.empty((S, A, S), dtype=torch.float32, device='cpu')  # final result on CPU

        for s_start in range(0, S, chunk_size):
            s_end = min(s_start + chunk_size, S)
            chunk_len = s_end - s_start

            # Start with all ones in chunk (chunk, A, S)
            chunk_joint_probs = torch.ones((chunk_len, A, S), dtype=torch.float32, device=device)

            for agent_id in range(N):
                # Local maps for current chunk
                s_local = state_map[s_start:s_end, agent_id]     # (chunk,)
                a_local = action_map[:, agent_id]                # (A,)
                y_local = state_map[:, agent_id]                 # (S,)

                # Expand for broadcasting
                s_exp = s_local[:, None, None]   # (chunk, 1, 1)
                a_exp = a_local[None, :, None]   # (1, A, 1)
                y_exp = y_local[None, None, :]   # (1, 1, S)

                # Transition probabilities: (chunk, A, S)
                agent_trans = transition_matrices[agent_id][s_exp, a_exp, y_exp]  # fancy indexing

                # Multiply into joint probabilities
                chunk_joint_probs *= agent_trans

            # Copy chunk to CPU result tensor
            joint_probs[s_start:s_end, :, :] = chunk_joint_probs.to('cpu')

        # Optionally save as numpy
        self.joint_transition_matrix = joint_probs.numpy().squeeze()
        self.transition_matrices = np.array(self.joint_transition_matrix).squeeze()


        # print(self.joint_transition_matrix.shape, self.transition_matrices.shape)

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
                    

    def run_MApolicy_cpu(self, policy, n_steps):
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

        if isinstance(policy, torch.Tensor):
            policy = policy.detach().cpu().numpy()
        
        for i in range(n_steps):
            action_probs = policy[state, :]/np.sum(policy[state, :])  # gives idx of joint action
            # action_probs = np.zeros(policy[state,:].shape)
            # action_probs[np.argmax(policy[state,:])] = 1

            # selects an action based on the probability distribution
            action = np.random.choice(self.n_joint_actions, p=action_probs)

            # Apply action and get next state and reward
            # gets next state based on initial state and chosen action
            next_state = self.get_next_state_cpu(state, action)
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
    
    def run_MApolicy(self, policy, n_steps):
        """
        GPU version of run_MApolicy using PyTorch tensors.
        Assumes get_next_state and joint_rewards are GPU-friendly.
        """

        state = torch.as_tensor(self.s0, dtype=torch.long, device=self.device).view(1)
        state_traj = torch.zeros(n_steps, dtype=torch.long, device=self.device)
        action_traj = torch.zeros(n_steps, dtype=torch.long, device=self.device)
        reward_traj = torch.zeros(n_steps, dtype=torch.float32, device=self.device)

        for i in range(n_steps):
            # Policy indexing stays on GPU
            action_probs = policy.index_select(0, state)  # (1, n_actions)
            action_probs = action_probs / action_probs.sum(dim=1, keepdim=True)

            # Sample action
            action = torch.multinomial(action_probs, num_samples=1).view(1)   # (1,)
            
            # Transition
            probs = self.joint_transition_matrix_t.index_select(0, state).index_select(1, action)

            probs = probs.squeeze(0).squeeze(0)  # (n_states,)
            next_state = torch.multinomial(probs, num_samples=1)  # (1,)

            # Reward
            reward = self.joint_rewards_t.index_select(0, state).index_select(1, action).squeeze()

            # Store
            state_traj[i] = state
            action_traj[i] = action
            reward_traj[i] = reward

            # Update state
            state = next_state

        reward_final = reward_traj.sum()
        return state_traj, action_traj, reward_final