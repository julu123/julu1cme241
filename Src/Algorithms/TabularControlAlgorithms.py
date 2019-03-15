from Algorithms.TabularBase import TabularBase
import numpy as np
from Processes.Variables import State, Action, Policy, Transitions_Rewards_Action_B

# All my tabular methods are slightly simplified. I didn't really understand what generalizations needed to be
# done in advance. The methods do work though.


class ControlMethods(TabularBase):
    def __init__(self,
                 mdp: Transitions_Rewards_Action_B,
                 gamma: float = 0.99):
        TabularBase.__init__(self, mdp, gamma)
        self.gamma = gamma
        self.states = list(mdp)

    def sarsa_on_policy(self,
                        pol: Policy,
                        alpha: float = 0.1,
                        episode_size: int = 500,
                        nr_episodes: int = 1000,
                        print_text: bool = False):

        q0 = {state: {action: 0 for action in pol[state]} for state in self.states}
        for i in range(nr_episodes):
            sim_states, sim_action, sim_rewards = self.generate(pol, steps=episode_size, print_text=print_text)
            for j in range(len(sim_states)-1):
                current_state = sim_states[j]
                current_action = sim_action[j]

                next_state = sim_states[j + 1]
                next_action = sim_action[j + 1]

                if self.investigate_termination(current_state, pol) is True:
                    reward = 0
                else:
                    reward = sim_rewards[j]

                q0[current_state][current_action] = q0[current_state][current_action] + \
                                                    alpha * (reward + q0[next_state][next_action]
                                                             - q0[current_state][current_action])
        return q0

    def q_learning_off_policy(self,
                              zero_pol: Policy, #this is needed in order to incorporporate which movies are possible at all states
                              alpha: float = 0.1,
                              epsilon: float = 0.01,
                              start_state: State = None,
                              episode_size: int = 500,
                              nr_episodes: int = 1000):
        q0 = {state: {action: 0 for action in zero_pol[state]} for state in self.states} #values
        pi0 = {state: {action: 1/len(zero_pol[state]) for action in zero_pol[state]} for state in self.states} #policy
        for i in range(nr_episodes):
            if start_state is None:
                current_state = np.random.choice(self.states)
            else:
                current_state = start_state
            current_action = np.random.choice(list(zero_pol[current_state]))
            for j in range(episode_size):
                sim_states, sim_dist, sim_rewards = self.generate_one_step_dist(current_state, current_action)
                "find out next step"
                next_state = np.random.choice(sim_states, replace=True, p=sim_dist)
                "find out next reward"
                reward = sim_rewards[sim_states.index(next_state)]
                "find possible actions"
                next_action_list = list(zero_pol[next_state])
                m = len(next_action_list)
                "find action with maximum value"
                initial_action_value = - np.inf
                for action in next_action_list:
                    if q0[next_state][action] > initial_action_value:
                        initial_action_value = q0[next_state][action]
                        max_action = action
                action_dist = []
                for action in next_action_list:
                    if action == max_action:
                        pi0[next_state][action] = 1 - epsilon
                    else:
                        pi0[next_state][action] = epsilon/(m-1)
                    action_dist.append(pi0[next_state][action])

                next_action = np.random.choice(next_action_list, replace=True, p=action_dist)
                q0[current_state][current_action] = q0[current_state][current_action] \
                                                    + alpha * (reward + self.gamma * q0[next_state][next_action]
                                                               - q0[current_state][current_action])
                current_action = next_action
                current_state = next_state
        return q0, pi0

    def sarsa_lambda(self,
                     pol: Policy,
                     alpha: float = 0.1,
                     lambd: float = 0.5,
                     start_state: State = None,
                     episode_size: int = 500,
                     nr_episodes: int = 1000,
                     print_text: bool = False):
        q0 = {state: {action: 0 for action in pol[state]} for state in self.states}  # values
        for i in range(nr_episodes):
            e_trace = {state: {action: 0 for action in pol[state]} for state in list(pol)}
            sim_states, sim_action, sim_rewards = self.generate(pol,
                                                                state=start_state,
                                                                steps=episode_size,
                                                                print_text=print_text)
            for j in range(len(sim_states) - 1):

                current_state = sim_states[j]
                current_action = sim_action[j]

                next_state = sim_states[j + 1]
                next_action = sim_action[j + 1]

                if self.investigate_termination(current_state, pol) is True:
                    reward = 0
                else:
                    reward = sim_rewards[j]

                delta = reward + self.gamma * q0[next_state][next_action] - q0[current_state][current_action]
                e_trace[current_state][current_action] += 1
                for state in list(pol):
                    for action in list(pol[state]):
                        q0[state][action] = q0[state][action] + alpha * e_trace[state][action] * delta
                        e_trace[state][action] = self.gamma * lambd * e_trace[state][action]
        return q0
