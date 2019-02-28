import numpy as np
from Variables import State, Action, SA, SR
from MDP_B import MDP_B
from Variables import State, Action, Policy, Transitions_Rewards_Action_B
import random
import matplotlib.pyplot as plt


class TabularBase(MDP_B):  # This is just a simple way to generate data for test in MC
    def __init__(self, mdp: Transitions_Rewards_Action_B, gamma: float = 0.99):
        MDP_B.__init__(self, mdp, gamma)

    def generate(self, pol: Policy, state: State = None, steps: int = 10, print_text: bool = False):
        return self.generate_path(pol, state, steps, print_text)


class PredictionMethods(TabularBase):
    def __init__(self,
                 mdp: Transitions_Rewards_Action_B,
                 pol: Policy,
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
                  nr_episodes: int = 1000,
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
                    for n in range(1, len(sim_states) - 1 - t):
                        g_t = rewards[t]
                        for k in range(1, n + 1):
                            g_t = g_t + self.gamma * rewards[t + k]
                        g_t_lambda += lambd ** (n - 1) * g_t
                    g_t_lambda = (1 - lambd) * g_t_lambda
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
                g_t_lambda = 0
                for t in range(1, len(sim_states) - 1):
                    g_t = rewards[0]
                    for n in range(1, t):
                        g_t = g_t + self.gamma**n * rewards[n]
                    g_t_lambda = g_t_lambda + g_t * lambd**(t-1)
                g_t_lambda = g_t_lambda*(1-lambd)
                v0[sim_states[0]] = v0[sim_states[0]] + alpha * (g_t_lambda - v0[sim_states[0]])
        elif method == "Backward" and update == "Offline":
            pass
        return v0