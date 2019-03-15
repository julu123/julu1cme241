import numpy as np
from Processes.Variables import State


def epsilon_greedy(self,
                   discrete_indicator: bool,
                   feasible_actions,
                   state: State,
                   weights: np.ndarray,
                   epsilon: float = 0.1):
    # If the action list is finite we can use a simple for loop
    # Else we can differentiate phi(s,a) * w wrt a and choose maximum a - for now I am using a loop here as well..
    if discrete_indicator is True:
        m = len(feasible_actions[state])
        optimal_value = - np.inf
        optimal_action = None
        for action in feasible_actions[state]:
            v_hat = np.matmul(self.get_features_w_action(state, action), weights)
            if v_hat > optimal_value:
                optimal_value = v_hat
                optimal_action = action
        discrete_distribution = []
        for action in feasible_actions[state]:
            if action == optimal_action:
                discrete_distribution.append(1 - epsilon + epsilon / m)
            else:
                discrete_distribution.append(epsilon /m)
        return np.random.choice(feasible_actions[state], p=discrete_distribution)
    elif discrete_indicator is False:
        # I am sure there is some more efficient way to do this, but this is fairly easy
        min_value, max_value = feasible_actions[state]
        space = 100  # hard coded for now
        possible_range = np.linspace(min_value, max_value, space)
        optimal_value = - np.inf
        optimal_action = None
        for action in possible_range:
            v_hat = np.matmul(self.get_features_w_action(state, action), weights)
            if v_hat > optimal_value:
                optimal_value = v_hat
                optimal_action = action
        cont_distribution = []
        for action in possible_range:
            if action == optimal_action:
                cont_distribution.append(1 - epsilon + epsilon / space)
            else:
                cont_distribution.append(epsilon / space)
        return np.random.choice(possible_range, p=cont_distribution)
