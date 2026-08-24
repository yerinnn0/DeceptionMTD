"""Single-agent 5x5 stochastic gridworld for DeceptionMTD.

This file is intended to replace the repository's ``MDP.py``.  It preserves
the public attributes and methods used by ``execute_serial.py``, the policy
optimizers, and the IRL classes, while fixing the environment to one agent,
25 states, and four actions.

Coordinates are ``(row, column)`` with ``(0, 0)`` at the upper-left.
Actions are UP=0, RIGHT=1, DOWN=2, LEFT=3.
"""

from __future__ import annotations

from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch


GridCoordinate = Tuple[int, int]

GRID_SHAPE = (5, 5)
START = (0, 0)
TERMINAL = (4, 4)

UP, RIGHT, DOWN, LEFT = range(4)
ACTION_NAMES = ("up", "right", "down", "left")
ACTION_DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))


def state_index(coord: GridCoordinate, grid_shape: Tuple[int, int] = GRID_SHAPE) -> int:
    """Convert ``(row, column)`` to a row-major integer state."""
    row, col = coord
    rows, cols = grid_shape
    if not (0 <= row < rows and 0 <= col < cols):
        raise ValueError(f"coordinate {coord} is outside grid {grid_shape}")
    return row * cols + col


def state_coordinate(state: int, grid_shape: Tuple[int, int] = GRID_SHAPE) -> GridCoordinate:
    """Convert a row-major integer state to ``(row, column)``."""
    rows, cols = grid_shape
    if not (0 <= int(state) < rows * cols):
        raise ValueError(f"state {state} is outside grid {grid_shape}")
    return divmod(int(state), cols)


def path_state_sets(
    grid_shape: Tuple[int, int] = GRID_SHAPE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(goal_states, preferred_states, decoy_states)``.

    ``goal_states`` contains the non-start upper/right path and the terminal.
    ``preferred_states`` is the same path without the shared terminal, so the
    equivocal term is not biased by a destination used by both routes.
    ``decoy_states`` contains the non-start left/lower path, also without the
    shared terminal.
    """
    rows, cols = grid_shape
    if rows < 2 or cols < 2:
        raise ValueError("grid must have at least two rows and two columns")

    true_coords = (
        [(0, col) for col in range(1, cols)]
        + [(row, cols - 1) for row in range(1, rows)]
    )
    preferred_coords = [coord for coord in true_coords if coord != (rows - 1, cols - 1)]
    decoy_coords = (
        [(row, 0) for row in range(1, rows)]
        + [(rows - 1, col) for col in range(1, cols - 1)]
    )

    goal_states = np.asarray([state_index(c, grid_shape) for c in true_coords], dtype=int)
    preferred_states = np.asarray(
        [state_index(c, grid_shape) for c in preferred_coords], dtype=int
    )
    decoy_states = np.asarray([state_index(c, grid_shape) for c in decoy_coords], dtype=int)
    return goal_states, preferred_states, decoy_states


def create_rewards(
    path_reward: float = 0.2,
    terminal_reward: float = 10.0,
    movement_cost: float = 0.1,
    grid_shape: Tuple[int, int] = GRID_SHAPE,
) -> np.ndarray:
    """Create true rewards with shape ``(n_states, 4)``.

    Rewards use the repository's current-state convention ``r(s, a)``.
    Every action receives ``movement_cost``; upper/right goal-path states
    receive ``path_reward``; and the terminal receives the additional
    ``terminal_reward`` once before it resets to the start.
    """
    n_states = int(np.prod(grid_shape))
    rewards = np.full((n_states, 4), float(movement_cost), dtype=np.float64)
    goal_states, _, _ = path_state_sets(grid_shape)
    rewards[goal_states, :] += float(path_reward)
    rewards[state_index((grid_shape[0] - 1, grid_shape[1] - 1), grid_shape), :] += float(
        terminal_reward
    )
    return rewards


def create_decoy_rewards(
    decoy_reward: float = 0.0,
    terminal_reward: float = 0.0,
    movement_cost: float = 0.0,
    grid_shape: Tuple[int, int] = GRID_SHAPE,
) -> np.ndarray:
    """Compatibility helper returning a decoy-route reward matrix.

    The single-agent experiment normally does not call this function.  It is
    retained because the original execution script imports it.
    """
    n_states = int(np.prod(grid_shape))
    rewards = np.full((n_states, 4), float(movement_cost), dtype=np.float64)
    _, _, decoy_states = path_state_sets(grid_shape)
    rewards[decoy_states, :] += float(decoy_reward)
    terminal = state_index((grid_shape[0] - 1, grid_shape[1] - 1), grid_shape)
    rewards[terminal, :] += float(terminal_reward)
    return rewards


def _as_flat_state_set(values: Iterable[int], n_states: int, name: str) -> np.ndarray:
    states = np.asarray(values, dtype=int).reshape(-1)
    if states.size and (np.any(states < 0) or np.any(states >= n_states)):
        raise ValueError(f"{name} contains a state outside [0, {n_states})")
    return np.unique(states)


class MultiAgentGridworld:
    """Drop-in-compatible single-agent stochastic gridworld.

    The original positional constructor is preserved.  For this environment,
    ``N_agents``, ``n_states``, and ``n_actions`` must be 1, 25, and 4.
    Additional keyword arguments expose the grid-specific settings.
    """

    def __init__(
        self,
        N_agents: int,
        initial_distribution: Sequence[float],
        n_states: int,
        n_actions: int,
        rewards: Optional[np.ndarray],
        gamma: float,
        v_reach: float = 0.9,
        variance: float = 0,
        p: float = 0.1,
        perturbation: int = 0,
        build_transition_matrix: bool = True,
        *,
        grid_shape: Tuple[int, int] = GRID_SHAPE,
        start: GridCoordinate = START,
        terminal: GridCoordinate = TERMINAL,
        chosen_action_probability: float = 0.9,
        random_action_probability: float = 0.1,
        path_reward: float = 1.0,
        terminal_reward: float = 0.0,
        movement_cost: float = 0.0,
        x_tar: Optional[np.ndarray] = None,
    ) -> None:
        del build_transition_matrix  # the 25x4x25 matrix is always cheap to build

        expected_states = int(np.prod(grid_shape))
        if N_agents != 1:
            raise ValueError("this replacement environment requires N_agents=1")
        if n_states != expected_states:
            raise ValueError(f"n_states must be {expected_states} for grid {grid_shape}")
        if n_actions != 4:
            raise ValueError("n_actions must be 4: up, right, down, left")
        if not 0 <= gamma < 1:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        if chosen_action_probability < 0 or random_action_probability < 0:
            raise ValueError("transition probabilities must be non-negative")
        if not np.isclose(chosen_action_probability + random_action_probability, 1.0):
            raise ValueError("chosen_action_probability + random_action_probability must equal 1")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.N_agents = 1
        self.n_states = expected_states
        self.n_actions = 4
        self.n_joint_states = expected_states
        self.n_joint_actions = 4
        self.grid_shape = tuple(grid_shape)
        self.start_coord = tuple(start)
        self.terminal_coord = tuple(terminal)
        self.start_state = state_index(self.start_coord, self.grid_shape)
        self.terminal_state = state_index(self.terminal_coord, self.grid_shape)
        self.action_names = ACTION_NAMES
        self.gamma = float(gamma)
        self.v_reach = float(v_reach)
        self.variance = variance
        self.p = p
        self.perturbation = perturbation
        self.chosen_action_probability = float(chosen_action_probability)
        self.random_action_probability = float(random_action_probability)

        initial = np.asarray(initial_distribution, dtype=np.float64).reshape(-1)
        if initial.size != self.n_states:
            raise ValueError(
                f"initial_distribution must contain {self.n_states} probabilities, got {initial.size}"
            )
        if np.any(initial < 0) or not np.isclose(initial.sum(), 1.0):
            raise ValueError("initial_distribution must be non-negative and sum to one")
        self.initial_distribution = initial.copy()
        self.initial_joint_distribution = initial.copy()
        self.initial_distribution_t = torch.as_tensor(
            self.initial_distribution, dtype=torch.float32, device=self.device
        )
        self.initial_joint_distribution_t = torch.as_tensor(
            self.initial_joint_distribution, dtype=torch.float32, device=self.device
        )
        self.s0 = int(np.random.choice(self.n_states, p=self.initial_distribution))

        goal, preferred, decoy = path_state_sets(self.grid_shape)
        if self.terminal_state not in goal:
            raise ValueError("terminal must be the lower-right endpoint of the configured true path")
        self.goal_states = goal
        self.preferred_states = preferred
        self.decoy_states = decoy
        # Apply the original reachability/task constraint to the complete true
        # path P_T.  goal_states already includes the shared terminal, while
        # preferred_states continues to exclude it for equivocal deception.
        self.task_states = goal.copy()
        self.true_path_states = goal.copy()

        if rewards is None:
            reward_matrix = create_rewards(
                path_reward=path_reward,
                terminal_reward=terminal_reward,
                movement_cost=movement_cost,
                grid_shape=self.grid_shape,
            )
        else:
            reward_matrix = np.asarray(rewards, dtype=np.float64)
            if reward_matrix.shape == (self.n_states, self.n_actions, 1):
                reward_matrix = reward_matrix[:, :, 0]
            if reward_matrix.shape != (self.n_states, self.n_actions):
                raise ValueError(
                    "rewards must have shape "
                    f"({self.n_states}, {self.n_actions}) or "
                    f"({self.n_states}, {self.n_actions}, 1)"
                )
        self.rewards = reward_matrix[:, :, None].copy()
        self.joint_rewards = reward_matrix.copy()

        self.build_joint_transition_matrix()
        self._sync_tensors()

        if x_tar is None:
            self.x_tar = self.build_target_occupancy_measure()
        else:
            self.set_target_occupancy_measure(x_tar)

    def _sync_tensors(self) -> None:
        self.joint_rewards_t = torch.as_tensor(
            self.joint_rewards, dtype=torch.float32, device=self.device
        )
        self.joint_transition_matrix_t = torch.as_tensor(
            self.joint_transition_matrix, dtype=torch.float32, device=self.device
        )

    def coord_to_state(self, coord: GridCoordinate) -> int:
        return state_index(coord, self.grid_shape)

    def state_to_coord(self, state: int) -> GridCoordinate:
        return state_coordinate(state, self.grid_shape)

    def set_goal_states(
        self,
        goal_states: Iterable[int],
        decoy_states: Iterable[int],
        preferred_states: Optional[Iterable[int]] = None,
    ) -> None:
        """Compatibility setter with an explicit preferred-set extension."""
        goals = _as_flat_state_set(goal_states, self.n_states, "goal_states")
        decoys = _as_flat_state_set(decoy_states, self.n_states, "decoy_states")
        preferred = goals[goals != self.terminal_state] if preferred_states is None else _as_flat_state_set(
            preferred_states, self.n_states, "preferred_states"
        )
        if self.terminal_state not in goals:
            raise ValueError("goal_states must include terminal_state")
        if self.terminal_state in preferred or self.terminal_state in decoys:
            raise ValueError("exclude the shared terminal from preferred_states and decoy_states")
        self.goal_states = goals
        self.preferred_states = preferred
        self.decoy_states = decoys
        self.task_states = goals.copy()
        self.true_path_states = goals.copy()

    def set_target_occupancy_measure(self, x_tar: np.ndarray) -> None:
        target = np.asarray(x_tar, dtype=np.float64)
        if target.size != self.n_joint_states * self.n_joint_actions:
            raise ValueError(
                f"x_tar must contain {self.n_joint_states * self.n_joint_actions} values"
            )
        target = target.reshape(self.n_joint_states, self.n_joint_actions)
        if np.any(target < 0) or not np.all(np.isfinite(target)):
            raise ValueError("x_tar must be finite and non-negative")
        self.x_tar = target.copy()

    def build_decoy_policy(self) -> np.ndarray:
        """Create a policy that prefers the left/lower route to the terminal."""
        policy = np.zeros((self.n_states, self.n_actions), dtype=np.float64)
        last_row, last_col = self.grid_shape[0] - 1, self.grid_shape[1] - 1
        for state in range(self.n_states):
            row, col = self.state_to_coord(state)
            if state == self.terminal_state:
                policy[state, :] = 1.0 / self.n_actions
            elif row < last_row and col == 0:
                policy[state, DOWN] = 1.0
            elif row == last_row and col < last_col:
                policy[state, RIGHT] = 1.0
            elif col > 0:
                policy[state, LEFT] = 1.0
            else:
                policy[state, DOWN] = 1.0
        return policy

    def build_target_occupancy_measure(
        self,
        policy: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Return a flow-feasible discounted occupancy target.

        Passing a policy lets the user define ``x_tar`` behaviorally.  With no
        policy, the target follows the decoy (left/lower) route.  A fully
        specified array can instead be installed with
        :meth:`set_target_occupancy_measure`.
        """
        if policy is None:
            policy = self.build_decoy_policy()
        policy = np.asarray(policy, dtype=np.float64)
        if policy.shape != (self.n_states, self.n_actions):
            raise ValueError(f"policy must have shape ({self.n_states}, {self.n_actions})")
        if np.any(policy < 0) or not np.allclose(policy.sum(axis=1), 1.0):
            raise ValueError("each policy row must be non-negative and sum to one")

        p_pi = np.einsum("sa,sak->sk", policy, self.transition_matrices)
        discounted_state_occupancy = np.linalg.solve(
            np.eye(self.n_states) - self.gamma * p_pi.T,
            self.initial_joint_distribution,
        )
        return discounted_state_occupancy[:, None] * policy

    def build_agent_transition_matrix(self, p: float = 0.1) -> np.ndarray:
        """Build ``T[s, a, s']`` with 0.9 chosen + 0.1 uniform actions.

        ``p`` is accepted for call compatibility; probabilities come from the
        constructor.  Boundary collisions add their probability to staying in
        place.  Every terminal action deterministically resets to the start.
        """
        del p
        transitions = np.zeros((self.n_states, self.n_actions, self.n_states), dtype=np.float64)
        random_share = self.random_action_probability / self.n_actions

        for state in range(self.n_states):
            if state == self.terminal_state:
                transitions[state, :, self.start_state] = 1.0
                continue
            for chosen_action in range(self.n_actions):
                actual_action_probs = np.full(self.n_actions, random_share, dtype=np.float64)
                actual_action_probs[chosen_action] += self.chosen_action_probability
                for actual_action, probability in enumerate(actual_action_probs):
                    next_state = self._deterministic_next_state(state, actual_action)
                    transitions[state, chosen_action, next_state] += probability

        if not np.allclose(transitions.sum(axis=2), 1.0):
            raise RuntimeError("transition rows do not sum to one")
        self.agent_transition_matrix = transitions
        return transitions

    def _deterministic_next_state(self, state: int, action: int) -> int:
        row, col = self.state_to_coord(state)
        d_row, d_col = ACTION_DELTAS[int(action)]
        next_row = min(max(row + d_row, 0), self.grid_shape[0] - 1)
        next_col = min(max(col + d_col, 0), self.grid_shape[1] - 1)
        return self.coord_to_state((next_row, next_col))

    def build_joint_transition_matrix(self) -> None:
        self.joint_transition_matrix = self.build_agent_transition_matrix().copy()
        self.transition_matrices = self.joint_transition_matrix

    def build_joint_transition_matrix_cpu(self) -> None:
        self.build_joint_transition_matrix()
        self._sync_tensors()

    def build_joint_transition_matrix_original(self) -> None:
        self.build_joint_transition_matrix_cpu()

    def build_joint_rewards(self) -> None:
        reward_matrix = np.asarray(self.rewards, dtype=np.float64)
        if reward_matrix.shape == (self.n_states, self.n_actions, 1):
            reward_matrix = reward_matrix[:, :, 0]
        if reward_matrix.shape != (self.n_states, self.n_actions):
            raise ValueError("invalid rewards shape")
        self.joint_rewards = reward_matrix.copy()
        self._sync_tensors()

    def reset_initial_state(self) -> None:
        self.s0 = int(np.random.choice(self.n_states, p=self.initial_distribution))

    def joint_state_space(self) -> np.ndarray:
        return np.arange(self.n_joint_states)

    def agent_state_space(self) -> np.ndarray:
        return np.arange(self.n_states)

    def joint_action_space(self) -> np.ndarray:
        return np.arange(self.n_joint_actions)

    def agent_action_space(self) -> np.ndarray:
        return np.arange(self.n_actions)

    def get_agent_slice(self, agent_id: int) -> slice:
        if agent_id != 0:
            raise IndexError("single-agent environment has only agent 0")
        return slice(0, self.n_states)

    def get_joint_state(self, states: Sequence[int]) -> int:
        values = np.asarray(states, dtype=int).reshape(-1)
        if values.size != 1:
            raise ValueError("single-agent joint state must contain one local state")
        return int(values[0])

    def get_joint_action(self, actions: Sequence[int]) -> int:
        values = np.asarray(actions, dtype=int).reshape(-1)
        if values.size != 1:
            raise ValueError("single-agent joint action must contain one local action")
        return int(values[0])

    def get_state(self, joint_state: int) -> list[int]:
        return [int(joint_state)]

    def get_action(self, joint_action: int) -> list[int]:
        return [int(joint_action)]

    def get_next_state_cpu(self, joint_state: int, joint_action: int) -> int:
        return int(
            np.random.choice(
                self.n_joint_states,
                p=self.joint_transition_matrix[int(joint_state), int(joint_action)],
            )
        )

    def get_next_state(self, joint_state: torch.Tensor, joint_action: torch.Tensor) -> torch.Tensor:
        state = torch.as_tensor(joint_state, dtype=torch.long, device=self.device).reshape(-1)
        action = torch.as_tensor(joint_action, dtype=torch.long, device=self.device).reshape(-1)
        if state.numel() != 1 or action.numel() != 1:
            raise ValueError("get_next_state expects one state and one action")
        probabilities = self.joint_transition_matrix_t[state[0], action[0]]
        return torch.multinomial(probabilities, num_samples=1).squeeze(0)

    def feature_vector(self, i: int, feature_map: str = "identity") -> np.ndarray:
        """Return a state-action feature vector without multi-agent assumptions."""
        index = int(i)
        total = self.n_states * self.n_actions
        if not 0 <= index < total:
            raise IndexError(f"feature index must be in [0, {total})")
        state, action = divmod(index, self.n_actions)
        row, col = self.state_to_coord(state)

        if feature_map in {"identity", "state_action", "set_local"}:
            feature = np.zeros(total, dtype=np.float64)
            feature[index] = 1.0
            return feature
        if feature_map in {"state", "local_state"}:
            feature = np.zeros(self.n_states, dtype=np.float64)
            feature[state] = 1.0
            return feature
        if feature_map == "coord":
            action_one_hot = np.eye(self.n_actions, dtype=np.float64)[action]
            row_scale = max(self.grid_shape[0] - 1, 1)
            col_scale = max(self.grid_shape[1] - 1, 1)
            return np.concatenate(
                [
                    np.asarray([row / row_scale, col / col_scale], dtype=np.float64),
                    action_one_hot,
                    np.asarray(
                        [
                            state in self.preferred_states,
                            state in self.decoy_states,
                            state == self.terminal_state,
                            1.0,
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
        if feature_map == "path":
            action_one_hot = np.eye(self.n_actions, dtype=np.float64)[action]
            route = np.asarray(
                [
                    state in self.preferred_states,
                    state in self.decoy_states,
                    state == self.terminal_state,
                    state not in set(self.goal_states) | set(self.decoy_states),
                ],
                dtype=np.float64,
            )
            return np.concatenate([route, action_one_hot])
        if feature_map == "proxi":
            proximity = np.zeros(self.n_states, dtype=np.float64)
            for other_state in range(self.n_states):
                other_row, other_col = self.state_to_coord(other_state)
                proximity[other_state] = 1.0 / (abs(row - other_row) + abs(col - other_col) + 1.0)
            return np.concatenate([proximity, np.eye(self.n_actions)[action]])
        if feature_map in {"agent_controlled", "correlated"}:
            raise ValueError(
                f"feature_map={feature_map!r} encodes the old multi-agent cyber states; "
                "use 'identity', 'state', 'coord', 'path', or 'proxi'"
            )
        raise ValueError(f"unknown feature_map {feature_map!r}")

    def feature_matrix(self, feature_map: str = "identity") -> np.ndarray:
        return np.asarray(
            [self.feature_vector(i, feature_map) for i in range(self.n_states * self.n_actions)]
        )

    def run_MApolicy_cpu(
        self, policy: np.ndarray, n_steps: int
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        if isinstance(policy, torch.Tensor):
            policy = policy.detach().cpu().numpy()
        policy = np.asarray(policy, dtype=np.float64).reshape(self.n_states, self.n_actions)
        state = int(self.s0)
        state_traj = np.zeros(n_steps, dtype=int)
        action_traj = np.zeros(n_steps, dtype=int)
        reward_traj = np.zeros(n_steps, dtype=np.float64)
        for step in range(n_steps):
            action_probs = policy[state] / policy[state].sum()
            action = int(np.random.choice(self.n_actions, p=action_probs))
            state_traj[step] = state
            action_traj[step] = action
            reward_traj[step] = self.joint_rewards[state, action]
            state = self.get_next_state_cpu(state, action)
        return state_traj, action_traj, float(reward_traj.sum())

    def run_MApolicy(
        self, policy: torch.Tensor, n_steps: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_t = torch.as_tensor(policy, dtype=torch.float32, device=self.device).reshape(
            self.n_states, self.n_actions
        )
        state = torch.tensor(int(self.s0), dtype=torch.long, device=self.device)
        state_traj = torch.zeros(n_steps, dtype=torch.long, device=self.device)
        action_traj = torch.zeros(n_steps, dtype=torch.long, device=self.device)
        reward_traj = torch.zeros(n_steps, dtype=torch.float32, device=self.device)
        for step in range(n_steps):
            action_probs = policy_t[state] / policy_t[state].sum()
            action = torch.multinomial(action_probs, num_samples=1).squeeze(0)
            state_traj[step] = state
            action_traj[step] = action
            reward_traj[step] = self.joint_rewards_t[state, action]
            state = torch.multinomial(
                self.joint_transition_matrix_t[state, action], num_samples=1
            ).squeeze(0)
        return state_traj, action_traj, reward_traj.sum()


__all__ = [
    "ACTION_NAMES",
    "DOWN",
    "GRID_SHAPE",
    "LEFT",
    "MultiAgentGridworld",
    "RIGHT",
    "START",
    "TERMINAL",
    "UP",
    "create_decoy_rewards",
    "create_rewards",
    "path_state_sets",
    "state_coordinate",
    "state_index",
]
