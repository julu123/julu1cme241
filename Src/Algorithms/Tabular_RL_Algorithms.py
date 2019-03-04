import numpy as np
from Processes.Variables import State, Action, SA, SR
from Processes.MDP_B import MDP_B
from Processes.Variables import State, Action, Policy, Transitions_Rewards_Action_B
import random
import matplotlib.pyplot as plt


class TabularBase(MDP_B):  # This is just a simple way to generate data
    def __init__(self, mdp: Transitions_Rewards_Action_B, terminal_state: State = None, gamma: float = 0.99):
        MDP_B.__init__(self, mdp, gamma)
        self.info = mdp
        self.terminal_state = terminal_state

    def generate(self, pol: Policy, state: State = None, steps: int = 10, print_text: bool = False):
        return self.generate_path(pol, state, steps, print_text)

    def generate_one_step_dist(self, state: State, action: Action):
        return self.generate_state_dist(state, action)

    def generate_action_dist(self, state: State, pol: Policy):
        return self.genarate_action_dist(state, pol)

    def investigate_termination(self, state: State, pol: Policy):
        for action in pol[state]:
            try:
                trans_prob = self.info[state][action][state][0]
            except KeyError:
                trans_prob = 0
            if trans_prob + pol[state][action] == 2:
                return True


class PredictionMethods(TabularBase):
    def __init__(self,
                 mdp: Transitions_Rewards_Action_B,
                 pol: Policy = None,
                 gamma: float = 0.99):
        "It needs to take in an mdp in order to generate data. The prediction mehtods do not know the probabilities!"
        TabularBase.__init__(self, mdp, gamma)
        self.pol = pol
        self.gamma = gamma
        self.states = list(mdp)

    def monte_carlo_first_visit(self,
                                episode_size: int = 500,
                                nr_episodes: int = 200,
                                print_text: bool = False):
        random.seed(1)
        v0 = {i: 0 for i in self.states}
        g0 = v0.copy()
        for i in range(nr_episodes):
            sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
            g_t = rewards[0]
            for j in range(1, len(sim_states) - 1):
                g_t = g_t + self.gamma**j * rewards[j]
            v0[sim_states[0]] = v0[sim_states[0]] + g_t
            g0[sim_states[0]] += 1
        for i in v0:
            v0[i] = v0[i]/g0[i]
        return v0

    def td_zero(self,
                alpha: float = 0.1,
                episode_size: int = 500,
                nr_episodes: int = 200,
                print_text: bool = False):
        random.seed(1)
        v0 = {i: 0 for i in self.states}
        for i in range(nr_episodes):
            sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
            for j in range(len(sim_states)-1):
                current_state = sim_states[j]
                next_state = sim_states[j+1]
                v0[current_state] = v0[current_state] + alpha * (rewards[j] + self.gamma*v0[next_state] - v0[current_state])
        return v0

    def td_lambda(self,
                  alpha: float = 0.1,
                  lambd: float = 0.8,
                  episode_size: int = 500,
                  nr_episodes: int = 10000,
                  method: str = "Forward",
                  update: str = "Online",
                  print_text: bool = False):
        random.seed(1)
        v0 = {i: 0 for i in self.states}
        if method == "Forward" and update == "Online":
            for i in range(nr_episodes):
                sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
                "This method seems to be wrong"
                for t in range(len(sim_states) - 1):
                    g_t_lambda = 0
                    final_g_t = 0
                    for n in range(1, len(sim_states) - 1 - t):
                        g_t = rewards[t]
                        for k in range(1, n + 1):
                            g_t = g_t + self.gamma * rewards[t + k]
                        final_g_t = g_t
                        g_t_lambda += lambd ** (n - 1) * g_t
                    g_t_lambda = (1 - lambd) * g_t_lambda + lambd**(len(sim_states)-1) * final_g_t
                    v0[sim_states[t]] = v0[sim_states[t]] + alpha * (g_t_lambda - v0[sim_states[t]])
        elif method == "Backward" and update == "Online":
            for i in range(nr_episodes):
                sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
                e_trace = {i: 0 for i in self.states}
                for t in range(len(sim_states) - 1):
                    for s in self.states:
                        e_trace[s] = e_trace[s] * lambd
                    e_trace[sim_states[t]] += 1
                    current_state = sim_states[t]
                    next_state = sim_states[t + 1]
                    v0[current_state] = v0[current_state] + alpha * \
                                        (rewards[t] + self.gamma * v0[next_state] - v0[current_state]) * \
                                        e_trace[current_state]
        elif method == "Forward" and update == "Offline":
            for i in range(nr_episodes):
                sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
                #rewards.append(0) this can be added to make the code simpler, for now it just terminates before the zero..
                g_t_lambda = 0
                final_g_t = 0
                for t in range(1, len(sim_states) - 1):
                    g_t = rewards[0]
                    for n in range(1, t):
                        g_t = g_t + self.gamma**n * rewards[n]
                    final_g_t = g_t
                    g_t_lambda = g_t_lambda + g_t * lambd**(t-1)
                g_t_lambda = g_t_lambda*(1-lambd) + lambd**(len(sim_states)-1) * final_g_t
                v0[sim_states[0]] = v0[sim_states[0]] + alpha * (g_t_lambda - v0[sim_states[0]])
        elif method == "Backward" and update == "Offline":
            pass
        return v0


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
                              nr_episodes: int = 1000,
                              print_text: bool = False):
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