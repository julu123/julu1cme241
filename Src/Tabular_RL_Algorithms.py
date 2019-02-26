import numpy as np
from Variables import State, Action, SA, SR
from MDP_B import MDP_B
from Variables import State, Action, Policy, Transitions_Rewards_Action_B
import random


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

    def monte_carlo_first_visit(self, episode_size: int = 500, nr_episodes: int = 100, print_text: bool = False):
        random.seed(1)
        v0 = {i: 0 for i in self.states}
        g0 = v0.copy()
        for i in range(nr_episodes):
            sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
            g_t = 0
            for j in reversed(range(len(sim_states)-1)):
                g_t = g_t * self.gamma + rewards[j]
            v0[sim_states[0]] = v0[sim_states[0]] + g_t
            g0[sim_states[0]] += 1
        for i in v0:
            v0[i] = v0[i]/g0[i]
        return v0

    def td_zero(self,
                alpha: float = 0.1,
                episode_size: int = 500,
                nr_episodes: int = 100,
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
                  lambd: float = 0.9,
                  episode_size: int = 500,
                  nr_episodes: int = 100,
                  method: str = "Forward",
                  print_text: bool = False):
        random.seed(1)
        v0 = {i: 0 for i in self.states}
        for i in range(nr_episodes):
            sim_states, _, rewards = self.generate(self.pol, steps=episode_size, print_text=print_text)
            e_trace = {i: 0 for i in self.states}
            if method == "Forward":
                for t in range(len(sim_states)-1):
                    for s in self.states:
                        e_trace[s] = e_trace[s]*lambd
                    e_trace[sim_states[t]] += 1
                    current_state = sim_states[t]
                    next_state = sim_states[t + 1]
                    v0[current_state] = v0[current_state] + alpha * \
                                        (rewards[t] + self.gamma * v0[next_state] - v0[current_state])\
                                        * e_trace[current_state]
            elif method == "Backward":
                "Some thing is wrong here!"
                e_trace[sim_states[0]] += 1
                for t in reversed(range(1, len(sim_states)-1)):
                    for s in self.states:
                        e_trace[s] = e_trace[s]*lambd
                    e_trace[sim_states[t]] += 1
                    current_state = sim_states[t]
                    next_step = sim_states[t + 1]
                    delta_t = rewards[t] + self.gamma * v0[next_step] - v0[current_state]
                    v0[current_state] = v0[current_state] + alpha * delta_t * e_trace[current_state]
        return v0


