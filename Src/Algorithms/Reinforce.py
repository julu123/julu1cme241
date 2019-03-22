import numpy as np
from Algorithms.FunctionApproximationBase import FunctionApproximationBase
from typing import Callable, Tuple, Dict
from Processes.Variables import State, States, Action

# Note! This code is nor working properly -- disrecte and continious action/state spaces need to bee implemented sepratly 


class REINFORCE(FunctionApproximationBase):
    def __init__(self,
                 generator_function: Callable[[State, Action], Tuple[State, (float or int)]],
                 policy: Callable[[State], Action],
                 state_action_feature_function: Callable[[State, Action], np.ndarray],
                 terminal_states: Callable[[State], bool] = None,
                 gamma: float = 0.99,
                 nr_features: int = 5,
                 list_of_states: States = None,
                 dist_of_action: Dict[State, Dict[Action]] = None):
        FunctionApproximationBase.__init__(generator_function=generator_function,
                                           state_action_feature_function=state_action_feature_function,
                                           policy=policy,
                                           terminal_states=terminal_states)
        self.gamma = gamma
        self.nr_features = nr_features
        self.states = list_of_states
        self.action = dist_of_action

    def learn(self,
              starting_state: State,
              nr_episodes: int = 1000,
              alpha: float = 0.1,
              decay: bool = False,
              max_iterations: int = 500,
              cont_or_disc: str = "Normal"):
        initial_action = self.get_action(starting_state)
        theta = np.random.randn((self.nr_features, 1)) * 0.01
        for i in range(nr_episodes):
            current_state = starting_state
            current_action = initial_action
            if decay is True:
                learning_rate = alpha - alpha * i / nr_episodes
            else:
                learning_rate = alpha
            for j in range(max_iterations):
                G = 0
                for k in range(j, max_iterations):
                    next_state, reward = self.generate(current_state, current_action)
                    G += self.gamma**(k-j)*reward

                    next_features = self.get_features(current_state)
                    next_action = self.policy(theta, next_features, cont_or_disc)

                    current_action = next_action
                    current_state = next_state
                if cont_or_disc == "Normal":
                    # Assume sigma = 1
                    nabla_log_pi = current_action * np.matmul(self.get_features(current_state), theta) * \
                                   (self.get_features(current_state)).T
                elif cont_or_disc == "Softmax":
                    # the expected value should look at the probabilities and not at theta -- i.e. this is wrong for now
                    exp_value = sum([np.matmul(self.get_features(state), theta) for state in self.states])
                    nabla_log_pi = (self.get_features_w_action(current_state, current_action)).T - exp_value
                theta += learning_rate * self.gamma**j * G * nabla_log_pi
        return theta

    def policy(self,
               weights: np.ndarray,
               features: np.ndarray,
               current_state: State,
               cont_or_disc: str = "Normal",
               ):
        if cont_or_disc == "Normal":
            return np.random.randn() + np.matmul(features, weights)
        elif cont_or_disc == "Softmax":
            probdist = []
            for action in self.actions[current_state]:
                probdist.append(np.exp(np.matmul(self.get_features_w_action(current_state, action)), weights))
            the_sum = sum(probdist)
            for i, j in enumerate(probdist):
                probdist[i] = j/the_sum
            return np.random.choice(self.actions[current_state], p=probdist)
