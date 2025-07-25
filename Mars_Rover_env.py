'''
Simon Hirlaender
For the course Reinforcement Learning 2024 - 536.111 at PLUS University
This script defines a framework for simulating a simplified Mars Rover navigation task as a reinforcement learning environment, leveraging the Gymnasium API. It includes the implementation of a discrete environment class, a specific Mars Rover environment, and a wrapper to convert the Mars Rover environment from a Markov Decision Process (MDP) to a Markov Reward Process (MRP) by applying a fixed policy.

**Key Components of the Script:**

- **`categorical_sample` Function**: Utilizes probabilities to stochastically select an index, simulating the randomness in action outcomes.
- **`DiscreteEnv` Class**: Serves as a base class for discrete action and state space environments, encapsulating common functionalities like resetting the environment and stepping through actions.
- **`MarsRoverEnv` Class**: Extends `DiscreteEnv` to model a Mars Rover navigation scenario with specific dynamics, where the rover moves on a linear track with probabilistic movement outcomes and rewards based on position.
- **`MarsRoverMRPWrapper` Class**: Transforms the Mars Rover MDP into an MRP by enforcing a fixed policy, thus simplifying decision-making to observe state and reward dynamics under specific behaviors.
- **Simulation**: The script demonstrates how to use the environment with a predefined policy that always chooses to move the rover right. It simulates the environment's behavior over a specified number of steps, collects states across episodes, and visualizes the rover's progression over time on a plot.

**Functional Overview:**

1. **Environment Setup**: Defines a custom Mars Rover environment where the rover has to navigate a linear track with probabilistic outcomes for its actions (move left, move right, stay).
2. **Policy Enforcement**: Introduces a wrapper that applies a fixed policy to the environment, effectively converting the problem from an MDP, where actions are chosen based on a policy learned or specified by the agent, to an MRP, where outcomes are solely determined by the state transitions and rewards under the given policy.
3. **Visualization**: Simulates the environment under the fixed policy for a number of steps, tracking the state transitions across multiple episodes. It then plots these transitions, showing the rover's state over time and marking the beginning of new episodes, to illustrate the rover's journey and the effects of the applied policy.

**Educational Value**: This script is an educational tool for understanding the basics of reinforcement learning, specifically the concepts of MDPs, MRPs, and policy application in a controlled environment. It showcases how to model an environment, apply a fixed policy, and analyze the resulting state and reward dynamics through visualization.
'''

import numpy as np
import gymnasium as gym
from gymnasium.utils import seeding
from matplotlib import pyplot as plt


def categorical_sample(prob_n, np_random):
    """
    Samples an index from a categorical distribution defined by prob_n using the provided random generator.

    Args:
        prob_n (array-like): An array of probabilities representing a categorical distribution.
        np_random (np.random.Generator): A NumPy random generator for stochastic sampling.

    Returns:
        int: The sampled index based on the distribution.
    """
    prob_n = np.asarray(prob_n)
    csprob_n = np.cumsum(prob_n)
    return (csprob_n > np_random.random()).argmax()


class DiscreteEnv(gym.Env):
    """
    A generic environment class for discrete state and action spaces.

    Attributes:
        nS (int): Number of states.
        nA (int): Number of actions.
        P (dict): Transitions dict of dicts of lists, where P[s][a] == [(probability, nextstate, reward, done), ...].
        isd (array-like): Initial state distribution.
    """

    def __init__(self, nS, nA, P, isd):
        self.P = P
        self.isd = isd
        self.nS = nS
        self.nA = nA
        self.action_space = gym.spaces.Discrete(nA)
        self.observation_space = gym.spaces.Discrete(nS)
        self.seed()
        self.reset()

    def seed(self, seed=None):
        """Seeds the environment's random number generator."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        """Resets the environment to an initial state."""
        self.s = categorical_sample(self.isd, self.np_random)
        return self.s

    def step(self, action):
        """Takes an action in the environment."""
        transitions = self.P[self.s][action]
        i = categorical_sample([t[0] for t in transitions], self.np_random)
        prob, next_state, reward, done = transitions[i]
        self.s = next_state
        return (next_state, reward, done, {"prob": prob})


class MarsRoverEnv(DiscreteEnv):
    """
    Mars Rover Environment, a specific instance of a discrete environment with custom dynamics.

    Args:
        n_states (int): Number of states excluding terminal states.
        p_stay (float): Probability of staying in the same state.
        p_backward (float): Probability of moving backward.
        left_side_reward (float): Reward for reaching the left terminal state.
        right_side_reward (float): Reward for reaching the right terminal state.
    """
    LEFT, RIGHT = 0, 1  # Define action constants for clarity

    def __init__(self, n_states=5, p_stay=1 / 3, p_backward=1 / 3, left_side_reward=1, right_side_reward=10):
        self.nS = n_states + 2  # Account for terminal states
        self.nA = 2  # Two actions: Left and Right
        isd = np.zeros(self.nS)
        isd[n_states // 2] = 1.0  # Start in the middle state
        P = self._create_transition_matrix(n_states, p_stay, p_backward, left_side_reward, right_side_reward)
        super().__init__(self.nS, self.nA, P, isd)

    def _create_transition_matrix(self, n_states, p_stay, p_backward, left_side_reward, right_side_reward):
        """
        Creates the transition probability matrix for the Mars Rover environment.
        """
        P = {s: {a: [] for a in range(self.nA)} for s in range(self.nS)}
        for s in range(1, self.nS - 1):
            for a in [self.LEFT, self.RIGHT]:
                outcomes = self._calculate_outcomes(s, a, p_stay, p_backward, left_side_reward, right_side_reward)
                P[s][a] = outcomes

        return P

    def _calculate_outcomes(self, s, action, p_stay, p_backward, left_side_reward, right_side_reward):
        """
        Calculate the possible outcomes of taking an action in state s.

        Args:
            s (int): Current state index.
            action (int): Action taken, either LEFT or RIGHT.
            p_stay (float): Probability of staying in the same state.
            p_backward (float): Probability of moving backward.
            left_side_reward (float): Reward received when reaching the leftmost state.
            right_side_reward (float): Reward received when reaching the rightmost state.

        Returns:
            list of tuples: Each tuple contains (probability, next_state, reward, done_flag).
        """
        # Calculate the probability of moving forward
        p_forward = 1.0 - p_stay - p_backward

        # Determine the direction of movement based on action
        move = -1 if action == self.LEFT else 1

        # Calculate forward and backward states, ensuring they are within bounds
        s_forward = np.clip(s + move, 0, self.nS - 1)
        s_backward = np.clip(s - move, 0, self.nS - 1)

        # Dictionary for rewards based on reaching terminal states
        reward_for_state = {0: left_side_reward, self.nS - 1: right_side_reward}

        # Calculate rewards and done flags for forward and backward moves
        r_forward = reward_for_state.get(s_forward, 0)
        r_backward = reward_for_state.get(s_backward, 0)

        # Check if forward or backward states are terminal
        d_forward = s_forward in (0, self.nS - 1)
        d_backward = s_backward in (0, self.nS - 1)

        # Construct and return the list of outcomes
        outcomes = [
            (p_forward, s_forward, r_forward, d_forward),
            (p_stay, s, 0.0, s in (0, self.nS - 1)),
            (p_backward, s_backward, r_backward, d_backward)
        ]

        return outcomes

    def render(self, mode='human'):
        # Implement rendering if needed
        pass


class MarsRoverMRPWrapper:
    """
    A wrapper to convert MarsRoverEnv MDP into an MRP by applying a fixed policy.

    Attributes:
        env (MarsRoverEnv): The environment to wrap.
        policy (dict): A dictionary mapping from state indices to action indices,
                       defining the fixed policy to apply.
    """

    def __init__(self, env, policy):
        """
        Initializes the wrapper with the environment and the policy.

        Args:
            env (MarsRoverEnv): The Mars Rover environment instance.
            policy (dict): The fixed policy, mapping from state to action.
        """
        self.env = env
        self.policy = policy

    def reset(self):
        """Resets the environment to its initial state."""
        return self.env.reset()

    def step(self):
        """
        Takes a step in the environment using the action defined by the fixed policy for the current state.

        Returns:
            A tuple of (next_state, reward, done, info), where 'info' contains the probability of the transition.
        """
        current_state = self.env.s
        action = self.policy[current_state]  # Get action from policy
        return self.env.step(action)

    # Optionally, implement any other methods from MarsRoverEnv you need, forwarding calls to self.env.


if __name__ == '__main__':
    # Initialize the original environment
    original_env = MarsRoverEnv()

    # Define a simple policy: always move right
    policy = {state: MarsRoverEnv.LEFT for state in range(original_env.nS)}

    # Wrap the environment with the defined policy
    MRP = MarsRoverMRPWrapper(original_env, policy)

    episodes = []  # List to hold lists of states for each episode
    rewards_episodes = []  # List to hold lists of rewards for each episode
    current_episode_states = []  # Temporarily hold states for the current episode
    current_episode_rewards = []  # Temporarily hold rewards for the current episode
    number_of_steps_with_environment = 100

    for _ in range(number_of_steps_with_environment):
        # action = MRP.env.action_space.sample()  # Random action
        state, reward, done, _ = MRP.step()
        current_episode_states.append(state)
        current_episode_rewards.append(reward)

        if done:
            episodes.append(current_episode_states)  # Add the completed episode states
            rewards_episodes.append(current_episode_rewards)  # Add the completed episode rewards
            current_episode_states = []  # Reset for next episode
            current_episode_rewards = []  # Reset rewards for next episode
            MRP.reset()

    # Check for any incomplete episodes and append them
    if current_episode_states:
        episodes.append(current_episode_states)
    if current_episode_rewards:
        rewards_episodes.append(current_episode_rewards)

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))  # Create 2 subplots

    # Plot states over time
    start_index = 0  # To keep track of where each episode starts
    for episode_states in episodes:
        axs[0].plot(range(start_index, start_index + len(episode_states)),
                    episode_states, marker='o', drawstyle='steps-post', color='blue')
        if len(episode_states) > 0:  # Check to ensure the episode is not empty
            axs[0].axvline(x=start_index, color='r', linestyle='--')  # Mark the start of each episode
            start_index += len(episode_states)  # Move the start index for the next episode

    axs[0].set_title('Mars Rover States Over Time')
    axs[0].set_xlabel('Step')
    axs[0].set_ylabel('State')
    y_ticks = np.arange(0, MRP.env.nS)
    y_labels = y_ticks + 1
    axs[0].set_yticks(y_ticks, y_labels)
    axs[0].grid(True)

    # Plot rewards over time
    start_index = 0
    for episode_rewards in rewards_episodes:
        axs[1].plot(range(start_index, start_index + len(episode_rewards)),
                    episode_rewards, marker='o', drawstyle='steps-post', color='green')
        if len(episode_rewards) > 0:
            axs[1].axvline(x=start_index, color='r', linestyle='--')
            start_index += len(episode_rewards)

    axs[1].set_title('Mars Rover Rewards Over Time')
    axs[1].set_xlabel('Step')
    axs[1].set_ylabel('Reward')
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()
