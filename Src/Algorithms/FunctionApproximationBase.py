from Processes.Variables import State, Action
import numpy as np
from typing import Callable, Tuple


class FunctionApproximationBase:

    def __init__(self,
                 generator_function: Callable[[State, Action], Tuple[State, (float or int)]],
                 policy: Callable[[State], Action],
                 feature_function: Callable[[State], np.ndarray] = None,
                 state_action_feature_function: Callable[[State, Action], np.ndarray] = None,
                 terminal_states: Callable[[State], bool] = None,
                 gamma: float = 0.99):
        self.gamma = gamma
        self.generator_function = generator_function
        self.feature_function = feature_function
        self.terminal_states = terminal_states
        self.state_action_feature_function = state_action_feature_function
        self.policy = policy

    def generate(self, state: State, action: Action):
        return self.generator_function(state, action)

    def get_features(self, state: State):
        return self.feature_function(state)

    def investigate_termination(self, state: State):
        return self.terminal_states(state)

    def get_action(self, state: State):
        return self.policy(state)

    def get_features_w_action(self, state: State, action: Action):
        return self.state_action_feature_function(state, action)
