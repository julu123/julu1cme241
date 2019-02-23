import numpy as np
from Variables import State, Action, SA, SR
from MDP_A import MDP_A
from MDP_B import MDP_B
from Variables import State, States, R_A, Transitions, Action, Transitions_Rewards_Action_A, Policy, Transitions_Rewards_Action_B


class SarsaGenerator(MDP_A or MDP_B):  # This is just a simple way to generate data for test in MC

    def __init__(self, mdp: (Transitions_Rewards_Action_A or Transitions_Rewards_Action_B), gamma: float = 1):
        MDP_A.__init__(self, mdp, gamma)

    def generate(self, pol: Policy, state: State = None, steps: int = 10):
        return self.generate_path(pol, state, steps)


class TabularBase:
    pass


class TabularMC(TabularBase):
    def __init__(self, pol: Policy, gamma: float = 0.99):
        self.pol = pol
        self.gamma = gamma

    def first_visit(self, data_generator, episode_size: int = 500, n: int = 500):
        "data_generator need to have a function called generate. In this case I just used my old mdp for this."
        states = set(data_generator.generate(self.pol)[0])
        v0 = {i: (0) for i in states}
        g0 = v0.copy()
        for i in range(n):
            states, _, rewards = data_generator.generate(self.pol, steps=episode_size)
            g_t = 0
            for j in reversed(range(len(states)-1)):
                g_t = g_t * self.gamma + rewards[j]
            v0[states[0]] = v0[states[0]] + g_t
            g0[states[0]] += 1
        for i in v0:
            v0[i] = v0[i]/g0[i]
        return v0