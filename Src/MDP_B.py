import numpy as np
from typing import TypeVar, Dict, List, Tuple
from MRP_B import MRP_B
from MDP_A import MDP_A
from Variables import State, States, Transitions, PR, Transitions_rewards_B, R_B, Action, Policy, Transitions_Rewards_Action_B


class MDP_B(MRP_B):
    def __init__(self, ProbDist: Transitions_Rewards_Action_B = None, gamma: float = 0.99):
        self.States=list(ProbDist)
        self.gamma=gamma
        self.all_info=ProbDist
        actions=[]
        for i in P:
            for j in P[i]:
                actions.append(j)
        self.Actions=sorted(list(set(actions)))
        self.Actiondict={i:{k for k in self.all_info[i]} for i in self.all_info}
        
    def get_MRP(self,pol: Policy):
        ProbDist=np.zeros((len(self.States),len(self.States)))
        rew=np.zeros((len(self.States),len(self.States)))
        for i in self.all_info:
            for j in self.all_info[i]:
                for k in self.all_info[i][j]:
                    ProbDist[list(self.all_info).index(i)][list(self.all_info).index(k)] \
                        += (self.all_info[i][j][k][0]*pol[i][j])
                    rew[list(self.all_info).index(i)][list(self.all_info).index(k)] \
                        += (self.all_info[i][j][k][1]*pol[i][j])
        return MRP_B(ProbDist, self.gamma, rew, self.States)
    
    def convert_to_A(self):
        ProbDist_A= {i: {j: ({k: self.all_info[i][j][k][0] for k in self.all_info[i][j]},
                           sum(list({self.all_info[i][j][k][0]*self.all_info[i][j][k][1]
                                     for k in self.all_info[i][j]})))
                        for j in self.all_info[i]}
                     for i in self.all_info}
        return MDP_A(ProbDist_A, self.gamma)

    def get_optimal_value_function(self, easy: bool = False, n: int = 100, threshold: float = 1e-3): #This is Value the Iteration
        V0 = dict([(s,0) for s in self.States])
        if easy is False:
            while True:
                Vk = V0.copy()
                delta = 0
                for s in self.States:
                    V0[s] = max([sum([(self.all_info[s][a][k][1] + Vk[k]*self.gamma)*self.all_info[s][a][k][0]
                                               for k in self.all_info[s][a]])
                                               for a in self.all_info[s]])
                    delta = max(delta, abs(Vk[s]-V0[s]))
                if delta < threshold*(1-self.gamma)/self.gamma:
                    break
            return Vk
        else:
            for i in range(n):
                Vk = V0.copy()
                for s in self.States:
                    V0[s] = max([sum([(self.all_info[s][a][k][1] + Vk[k]*self.gamma)*self.all_info[s][a][k][0]
                                               for k in self.all_info[s][a]])
                                               for a in self.all_info[s]])
            return Vk

    def get_optimal_policy(self, easy: bool = False, n: int = 100, threshold: float = 1e-3):
        pol = {i: {j: 1 / len(self.States) for j in self.all_info[i]} for i in self.all_info}
        v0 = dict([(s, 0) for s in self.States])
        pi = pol
        if easy is True:
            for i in range(n):
                vk = v0.copy()
                for s in self.States:
                    actlist = dict([(a, 0) for a in self.all_info[s]])
                    for a in self.all_info[s]:
                        for j in self.all_info[s][a]:
                            actlist[a] += self.all_info[s][a][j][0] * (self.all_info[s][a][j][1] + vk[j] * self.gamma)
                    pi[s] = max(actlist, key=actlist.get)
                    v0[s] = actlist[pi[s]]
            return {i: {j: (1 if pi[i] == j else 0) for j in self.all_info[i]} for i in self.all_info}
        else:
            value_function = self.get_optimal_value_function()
            while True:
                vk = v0.copy()
                delta = 0
                for s in self.States:
                    actlist = dict([(a, 0) for a in self.all_info[s]])
                    for a in self.all_info[s]:
                        for j in self.all_info[s][a]:
                            actlist[a] += self.all_info[s][a][j][0] * (self.all_info[s][a][j][1] + vk[j] * self.gamma)
                    pi[s] = max(actlist, key=actlist.get)
                    v0[s] = actlist[pi[s]]
                    delta = max(delta, abs(vk[s] - v0[s]))
                if delta < threshold * (1 - self.gamma) / self.gamma and sum([(v0[s] - value_function[s]) ** 2 for s in
                                                                              self.States]) < threshold:
                    break
            return {i: {j: (1 if pi[i] == j else 0) for j in self.all_info[i]} for i in self.all_info}

    def policy_evaluation(self,
                          pol: Policy,
                          easy: bool = False,
                          threshold: float = 1e-4):
        mrp = self.get_MRP(pol)
        if easy is True:
            print('method not yet developed. lol')
        else:
            v0 = dict([(s, 0) for s in self.States])
            #for i in range(1000):
            while True:
                vk = v0.copy()
                delta = 0
                for s in self.States:
                    v0[s] = sum([mrp.ProbDist[self.States.index(s)][self.States.index(k)]
                                 * (mrp.Rewards[self.States.index(s)][self.States.index(k)] + self.gamma * vk[k])
                                 for k in self.States])
                    delta = max(delta, abs(vk[s]-v0[s]))
                if delta < threshold*(1-self.gamma)/self.gamma:
                    break
        return vk

    def genarate_action_dist(self, state: State, pol: Policy):
        possible_actions = []
        possible_actions_probs = []
        for i in pol[state]:
            possible_actions.append(i)
            possible_actions_probs.append(pol[state][i])
        return possible_actions, possible_actions_probs

    def generate_state_dist(self, state: State, action: Action):
        possible_states = []
        possible_states_probs = []
        possible_rewards = []
        for s in self.all_info[state][action]:
            possible_states.append(s)
            possible_states_probs.append(self.all_info[state][action][s][0])
            possible_rewards.append(self.all_info[state][action][s][1])
        return possible_states, possible_states_probs, possible_rewards

    def generate_path(self, pol: Policy, start: State = None, steps: int = 10, print_text: bool = False):
        if start == None:
            start = np.random.choice(self.States)
        states = [start]
        actions = []
        rewards = []
        current_state = start
        i = 0
        while i <= steps:
            i += 1
            action_list, actions_dist = self.genarate_action_dist(current_state, pol)
            act = np.random.choice(action_list, replace=True, p=actions_dist)
            actions.append(act)

            states_list, states_dist, reward_list = self.generate_state_dist(current_state, act)
            state = np.random.choice(states_list, replace=True, p=states_dist)
            states.append(state)

            rew = reward_list[states_list.index(state)]
            rewards.append(rew)

            current_state = state
            path_counter = 0
            for a in self.all_info[current_state]:
                for s in self.all_info[current_state][a]:
                    if s == current_state:
                        path_counter += self.all_info[current_state][a][s][0] * pol[current_state][a]
            if path_counter == 1:
                i = steps + 1
                if print_text is True:
                    print('Termination at state:', current_state)
        return states, actions, rewards


#Test -- (Same Policy as for MDP_A)
P:Transitions_Rewards_Action_B={
    'Food': {
        'a': {'Food': (0.4, 1), 'Game': (0.6, 2)},
        'b': {'Food': (1/3, 3), 'Sleep' :(1/3, 4), 'Game': (1/3, 5)},
        'c': {'Sleep': (1, 6)}
    },
    'Sleep': {
        'a': {'Sleep': (0.2, 7), 'Game': (0.8, 9)},
        'c': {'Sleep': (0.7, 9), 'Game': (0.3, 10)}
    },
    'Game': {
        'a': {'Food': (0.9, -1), 'Sleep': (0.1, -2)},
        'b': {'Food': (1/3, -3), 'Sleep': (1/3, -4), 'Game': (1/3, -5)}
    }
}
