import numpy as np
from Variables import State, Action, SA, SR
from MDP_B import MDP_B
from Variables import State, States, R_A, Transitions, Action, Policy, Transitions_Rewards_Action_B


class TabularBase(MDP_B):  # This is just a simple way to generate data for test in MC
    def __init__(self, mdp: Transitions_Rewards_Action_B, gamma: float = 0.99):
        MDP_B.__init__(self, mdp, gamma)

    def generate(self, pol: Policy, state: State = None, steps: int = 10):
        return self.generate_path(pol, state, steps)


class TabularMC(TabularBase):
    def __init__(self, mdp: Transitions_Rewards_Action_B, pol: Policy, gamma: float = 0.99):
        "It needs to take in an mdp in order to generate data. The first visit model does not know the probabilities!"
        TabularBase.__init__(self, mdp, gamma)
        self.pol = pol
        self.gamma = gamma
        self.states = list(mdp)

    def first_visit(self, episode_size: int = 500, n: int = 500):
        v0 = {i: (0) for i in self.states}
        g0 = v0.copy()
        for i in range(n):
            sim_states, _, rewards = self.generate(self.pol, steps=episode_size)
            g_t = 0
            for j in reversed(range(len(sim_states)-1)):
                g_t = g_t * self.gamma + rewards[j]
            v0[sim_states[0]] = v0[sim_states[0]] + g_t
            g0[sim_states[0]] += 1
        for i in v0:
            v0[i] = v0[i]/g0[i]
        return v0