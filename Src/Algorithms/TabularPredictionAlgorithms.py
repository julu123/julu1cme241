import numpy as np
import random
from Algorithms.TabularBase import TabularBase
from Processes.Variables import State, Action, Policy, Transitions_Rewards_Action_B

# All my tabular methods are slightly simplified. I didn't really understand what generalizations needed to be
# done in advance. The methods do work though.


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
            if g0[i] != 0:
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
                  alpha: float = 0.05,
                  lambd: float = 0.8,
                  episode_size: int = 500,
                  nr_episodes: int = 100000,
                  method: str = "Forward",
                  update: str = "Online",
                  print_text: bool = False):
        v0 = {i: 0 for i in self.states}
        if method == "Forward" and update == "Online":
            vf_per_iterations = np.zeros((int(nr_episodes / 1), len(self.states)))
            for i in range(nr_episodes):
                sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
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
                    lr = alpha - alpha * i / nr_episodes
                    v0[sim_states[t]] = v0[sim_states[t]] + lr * (g_t_lambda - v0[sim_states[t]])
        elif method == "Backward" and update == "Online":
            vf_per_iterations = np.zeros((int(nr_episodes / 1), len(self.states)))
            for i in range(nr_episodes):
                sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
                e_trace = {i: 0 for i in self.states}
                for t in range(len(sim_states) - 1):
                    for s in self.states:
                        e_trace[s] = e_trace[s] * lambd
                    e_trace[sim_states[t]] += 1
                    current_state = sim_states[t]
                    next_state = sim_states[t + 1]
                    lr = alpha - alpha * i / nr_episodes
                    v0[current_state] = v0[current_state] + lr * \
                                        (rewards[t] + self.gamma * v0[next_state] - v0[current_state]) * \
                                        e_trace[current_state]
        elif method == "Forward" and update == "Offline":
            vf_per_iterations = np.zeros((int(nr_episodes/1), len(self.states)))
            for i in range(nr_episodes):
                sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
                g_t_lambda = 0
                final_g_t = 0
                for t in range(1, len(sim_states) - 1):
                    g_t = rewards[0]
                    for n in range(1, t):
                        g_t = g_t + self.gamma**n * rewards[n]
                    final_g_t = g_t
                    g_t_lambda = g_t_lambda + g_t * lambd**(t-1)
                g_t_lambda = g_t_lambda*(1-lambd) + lambd**(len(sim_states)-1) * final_g_t
                lr = alpha - alpha * i / nr_episodes
                v0[sim_states[0]] = v0[sim_states[0]] + lr * (g_t_lambda - v0[sim_states[0]])
                if (i+1) % 1 == 0:
                    for j, k in enumerate(v0):
                        vf_per_iterations[int((i+1)/1-1), j] = v0[k]
        elif method == "Backward" and update == "Offline":
            pass
        return v0, vf_per_iterations
