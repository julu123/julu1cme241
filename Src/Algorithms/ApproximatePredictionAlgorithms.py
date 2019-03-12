import numpy as np
from typing import Callable, Tuple
from Algorithms.FunctionApproximationBase import FunctionApproximationBase
from Processes.Variables import State, Action


class ApproximatePredictionMethods(FunctionApproximationBase):

    def __init__(self,
                 generator_function: Callable[[State, Action], Tuple[State, (float or int)]],
                 feature_function: Callable[[State], np.ndarray],
                 policy: Callable[[State], Action],
                 terminal_states: Callable[[State], bool] = None,
                 gamma: float = 0.99):
        FunctionApproximationBase.__init__(generator_function=generator_function,
                                           feature_function=feature_function,
                                           policy=policy,
                                           terminal_states=terminal_states)
        self.gamma = gamma

    def initialize_w(self, features):
        size = features.shape
        size = max(size[0], size[1])
        return np.random.randn(size, 1)*0.01

    def learn(self,
              starting_state: State,
              nr_episodes: int = 10000,
              alpha: float = 0.1,
              decay: bool = False,
              max_iterations: int = 500,
              update: str = "Online"):
        starting_features = self.get_features(starting_state)
        w = self.initalize_w(starting_features)
        for i in range(nr_episodes):
            current_state = starting_state  # Perhaps we should be able to choose this randomly
            ticker = 0
            g_t = 0

            if decay is True:
                learning_rate = alpha - alpha * i / nr_episodes
            else:
                learning_rate = alpha

            while self.investigate_termination(current_state) is False and ticker < max_iterations:
                # Do an action
                current_action = self.policy(current_state)
                # Observe state and reward
                next_state, reward = self.generate(current_state, current_action)
                # Update G_t
                g_t += self.gamma**ticker*reward
                # Update ticker and state
                current_state = next_state
                ticker += 1
                if update == "Online":
                    gradient = self.get_features(current_state).T
                    v_hat = float(np.dot(self.get_features(current_state), w))
                    w += learning_rate * (g_t - v_hat) * gradient
            if update == "Offline":
                gradient = self.get_features(starting_state).T
                v_hat = float(np.dot(self.get_features(starting_state), w))
                w += learning_rate * (g_t - v_hat) * gradient
        return w

    def td_zero(self,
                starting_state: State,
                nr_episodes: int = 10000,
                alpha: float = 0.1,
                decay: bool = False,
                max_iterations: int = 500):
        starting_features = self.get_features(starting_state)
        w = self.initalize_w(starting_features)
        for i in range(nr_episodes):
            current_state = starting_state
            ticker = 0

            if decay is True:
                learning_rate = alpha - alpha * i / nr_episodes
            else:
                learning_rate = alpha

            while self.investigate_termination(current_state) is False and ticker < max_iterations:
                # Do an action
                current_action = self.policy(current_state)
                # Observe next state and reward
                next_state, reward = self.generate(current_state, current_action)
                # Update -- gradient is just transpose of current features since we use a linear function -- Hard coded
                gradient = self.get_features(current_state).T
                td_error = reward + np.matmul((self.gamma * self.get_features(next_state) -
                                               self.get_features(current_state)), w)
                w += learning_rate * td_error * gradient
                current_state = next_state
        return w

    def td_lambda_w_eligibility_traces(self,
                                       starting_state: State,
                                       lambd: float = 0.8,
                                       nr_episodes: int = 10000,
                                       alpha: float = 0.1,
                                       decay: bool = False,
                                       max_iterations: int = 500):
        starting_features = self.get_features(starting_state)
        w = self.initalize_w(starting_features)
        k = starting_features.shape
        k = max(k[0], k[1])
        for i in range(nr_episodes):
            current_state = starting_state
            z = np.zeros((k, 1))
            ticker = 0
            if decay is True:
                learning_rate = alpha - alpha * i / nr_episodes
            else:
                learning_rate = alpha
            while self.investigate_termination(current_state) is False and ticker < max_iterations:
                # Do an action
                current_action = self.policy(current_state)
                # Observe next state and reward
                next_state, reward = self.generate(current_state, current_action)
                # Update Z
                gradient = self.get_features(current_state).T
                z = self.gamma * lambd * z + gradient
                # Update w
                td_error = reward + np.matmul(self.gamma*self.get_fetures(next_state) -
                                              self.get_features(current_state), w)
                w += learning_rate * td_error * z
                current_state = next_state
        return w