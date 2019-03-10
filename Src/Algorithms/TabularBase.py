from Processes.MDP_B import MDP_B
from Processes.Variables import State, States, Action, Policy, Transitions_Rewards_Action_B


class TabularBase(MDP_B):  # This is just a simple way to generate data
    def __init__(self, mdp: Transitions_Rewards_Action_B, terminal_state: (State or States) = None, gamma: float = 0.99):
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
