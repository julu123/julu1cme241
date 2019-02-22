import numpy as np
from Variables import State, Action
from MDP_A import MDP_A
from MDP_B import MDP_B

class Tabular_MC:
    def __init__(self, mdp: (MDP_A or MDP_B), pol, mdp_type: str = "A"):
        if mdp_type == "A":
            self.mdp = MDP_A(mdp)
        elif mdp_type == "B":
            self.mdp = MDP_B(mdp)
        self.pol = pol
        self.amount_of_states = len(mdp.States)

    def first_visit(self):
        pass