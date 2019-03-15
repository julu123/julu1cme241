import numpy as np
from typing import Callable, Tuple, Dict
from Algorithms.FunctionApproximationBase import FunctionApproximationBase
from Processes.Variables import State, Action
import Utils.utils

# Only works for linear models. More fancy models would require a different gradient and that is
# not incorporated in my code.


class ApproximateControlMethods(FunctionApproximationBase):
    def __init__(self,
                 generator_function: Callable[[State, Action], Tuple[State, (float or int)]],
                 policy: Callable[[State], Action],
                 state_action_feature_function: Callable[[State, Action], np.ndarray],
                 terminal_states: Callable[[State], bool] = None,
                 gamma: float = 0.99):
        FunctionApproximationBase.__init__(generator_function=generator_function,
                                           state_action_feature_function=state_action_feature_function,
                                           policy=policy,
                                           terminal_states=terminal_states)
        self.gamma = gamma

    def initialize_w(self, features):
        size = features.shape
        size = max(size[0], size[1])
        return np.random.randn(size, 1) * 0.01

    def sarsa(self,
              starting_state: State,
              nr_episodes: int = 1000,
              alpha: float = 0.1,
              decay: bool = False,
              max_iterations: int = 500):
        initial_action = self.get_action(starting_state)
        initial_features = self.get_features_w_action(starting_state, initial_action)
        w = self.initialize_w(initial_features)
        for i in range(nr_episodes):
            current_state = starting_state
            current_action = initial_action
            current_features = initial_features
            ticker = 0

            if decay is True:
                learning_rate = alpha - alpha * i / nr_episodes
            else:
                learning_rate = alpha

            while self.investigate(current_state) is False and ticker < max_iterations:
                next_state, reward = self.generte(current_state, current_action)
                next_action = self.get_action(next_state)
                next_features = self.get_features_w_action(next_state, next_action)

                td_error = reward + self.gamma * np.matmul(next_features - current_features, w)
                w += learning_rate * td_error * current_features.T

                current_state = next_state
                current_action = next_state
                current_features = next_features
        return w

    def sarsa_lambda(self,
                     starting_state: State,
                     lambd: float = 0.8,
                     nr_episodes: int = 1000,
                     alpha: float = 0.1,
                     decay: bool = False,
                     max_iterations: int = 500):
        initial_action = self.get_action(starting_state)
        initial_features = self.get_features_w_action(starting_state, initial_action)
        k = initial_features.shape
        k = max(k[0], k[1])
        w = self.initialize_w(initial_features)
        for i in range(nr_episodes):
            # Initialize state -- this could be changed so that it is initialized randomly --
            # -- depending on the underlying problem
            current_state = starting_state
            # Choose action
            current_action = self.get_action(current_state)
            # Initialize features (i.e. X in the algorithm)
            current_features = self.get_features_w_action(current_state, current_action)
            # Set utils
            z = np.zeros((k, 1))
            q_old = 0
            ticker = 0
            # Assert a learning rate
            if decay is True:
                learning_rate = alpha - alpha * i / nr_episodes
            else:
                learning_rate = alpha

            while self.investigate_termination(current_state) is False and ticker < max_iterations:
                next_state, reward = self.generate(current_state, current_action)
                next_action = self.get_action(next_state)
                next_features = self.get_features_w_action(next_state, next_action)
                current_q = np.matmul(current_features, w)
                next_q = np.matmul(next_features, w)
                td_error = reward + self.gamma * next_q - current_q
                z = self.gamma * lambd * z + (1 - learning_rate *
                                              self.gamma * lambd * np.matmul(current_features, z)) * current_features.T
                w = w + learning_rate * (td_error + current_q - q_old) \
                    * z - learning_rate * (current_q - q_old) * current_features.T
                q_old = next_q
                current_features = next_features
                current_action = next_action
                current_state = next_state
        return w

    def q_learning(self,
                   starting_state: State,
                   state_action_list: Dict[State, Dict[Action]] = None,
                   state_action_dist: Callable[[State], Tuple[float, float]] = None, #Tuple[float, float] correspondsto max an min range of action
                   discrete_actions: bool = True,
                   nr_episodes: int = 1000,
                   alpha: float = 0.1,
                   decay: bool = False,
                   max_iterations: int = 500,
                   epsilon: float = 0.1):
        initial_action = self.get_action(starting_state)
        initial_features = self.get_features_w_action(starting_state, initial_action)
        w = self.initialize_w(initial_features)
        for i in range(nr_episodes):
            current_state = starting_state
            current_action = initial_action
            current_features = initial_features
            ticker = 0

            if decay is True:
                learning_rate = alpha - alpha * i / nr_episodes
            else:
                learning_rate = alpha

            while self.investigate_termination(current_state) is False and ticker < max_iterations:
                next_state, reward = self.generate(current_state, current_action)
                if discrete_actions is True:
                    next_action = Utils.utils.epsilon_greedy(discrete_actions, next_state, state_action_list, w, epsilon)
                else:
                    next_action = Utils.utils.epsilon_greedy(discrete_actions, next_state, state_action_dist, w, epsilon)
                next_features = self.get_features_w_action(next_state, next_action)

                td_error = reward + np.matmul(self.gamma * next_features - current_features, w)
                w += learning_rate * td_error * current_features.T

                current_state = next_state
                current_action = next_state
                current_features = next_features
        return w