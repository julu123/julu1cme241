import numpy as np
from typing import TypeVar, Dict, List
from MRP_A import MRP_A
from Variables import State, States, R_A, Transitions, Action, Transitions_Rewards_Action_A, Policy

class MDP_A(MRP_A):
    def __init__(self, probdist: Transitions_Rewards_Action_A = None, gamma: float = 0.99):
        self.States=list(probdist)
        self.gamma=gamma
        self.all_info=probdist
        actions=[]
        for i in P:
            for j in P[i]:
                actions.append(j)
        self.Actions=sorted(list(set(actions)))
        self.Actiondict={i:{k for k in self.all_info[i]} for i in self.all_info}
    
    def get_MRP(self,Pol:Policy):
        ProbDist=np.zeros((len(self.States),len(self.States)))
        rew=np.zeros((len(self.States),1))
        for i in self.all_info:
            for j in self.all_info[i]:
                rew[list(self.all_info).index(i)] += self.all_info[i][j][1]*Pol[i][j]
                for k in self.all_info[i][j][0]:
                    ProbDist[list(self.all_info).index(i)][list(self.all_info).index(k)] += (self.all_info[i][j][0][k]*Pol[i][j])
        rewards = [float(i) for i in rew]
        return(MRP_A(ProbDist,self.gamma,rewards,self.States))

    def genarate_action_dist(self,state:State,Pol:Policy):
        possible_actions = []
        possible_actions_probs = []
        for i in Pol[state]:
            possible_actions.append(i)
            possible_actions_probs.append(Pol[state][i])
        return possible_actions, possible_actions_probs

    def generate_state_dist(self,state:State, action:Action):
        possible_states = []
        possible_states_probs = []
        for i in self.all_info[state][action][0]:
            possible_states.append(i)
            possible_states_probs.append(self.all_info[state][action][0][i])
        return possible_states, possible_states_probs, self.all_info[state][action][1]

    def generate_path(self, pol: Policy, start: State = None, steps: int = 10):
        if start == None:
            start = np.random.choice(self.States)
        States = [start]
        Actions = []
        Rewards = []
        current_state=start
        i=0
        while i <= steps:
            i+=1
            actions,act_dist=self.genarate_action_dist(current_state,pol)
            act = np.random.choice(actions, replace=True, p=act_dist)
            Actions.append(act)
            states,state_dist,rew = self.generate_state_dist(current_state,act)
            Rewards.append(rew)
            state = np.random.choice(states, replace = True, p=state_dist)
            States.append(state)
            current_state=state
        return States, Actions, Rewards

    def get_Optimal_Value_Function(self,treshold:float=1e-4): #This is Value the Iteration
        V0 = dict([(s,0) for s in self.States])
        while True:
            Vk = V0.copy()
            delta = 0
            for s in self.States:
                V0[s] = max(self.all_info[s][a][1]+sum([self.gamma*self.all_info[s][a][0][k]*Vk[k] 
                            for k in self.all_info[s][a][0]]) 
                            for a in self.all_info[s])
                delta = max(delta,abs(Vk[s]-V0[s]))
            if delta < treshold*(1-self.gamma)/self.gamma:
                break
        return Vk     
        
    def policy_Evaluation(self,
                          pol:Policy,
                          easy=False,
                          treshold: float = 1e-4): #Not final
        mrp=self.get_MRP(pol)
        if easy == True:
            return mrp.get_Value_Function()
        else: 
            V0 = dict([(s ,0) for s in self.States])
            #while True:
            for i in range(1000):
                Vk = V0.copy()
                delta = 0
                for s in self.States:
                    V0[s] = mrp.Rewards[self.States.index(s)] + self.gamma*sum([mrp.ProbDist[self.States.index(s)][self.States.index(k)]*Vk[k] for k in self.States])
                delta = max(delta,abs(Vk[s]-V0[s]))
                if delta < treshold*(1-self.gamma)/self.gamma:
                    break
        return Vk
    
    def get_Optimal_Policy(self, pol: Policy = None, Value_function = None, treshold: float = 1e-4):
        if pol == None:
            pol = {i:{j:1/len(self.States) for j in self.all_info[i]} for i in self.all_info}
        if Value_function == None:
            Value_function = self.Get_Optimal_Value_Function()
        V0 = dict([(s,0) for s in self.States])
        pi=pol
        while True:
            Vk=V0.copy()
            delta=0
            for s in self.States:
                actlist=dict([(a,0) for a in self.all_info[s]])
                #print(actlist)
                for a in self.all_info[s]:
                    test=0
                    for j in self.all_info[s][a][0]:
                        test += self.all_info[s][a][0][j]*(self.all_info[s][a][1]+self.gamma*Vk[j])
                        actlist[a] += self.all_info[s][a][0][j]*(self.all_info[s][a][1]+self.gamma*Vk[j])
                pi[s]=max(actlist, key=actlist.get)
                V0[s]=actlist[pi[s]]
                #print(actlist)
                #print(pi[s])
                delta = max(delta,abs(Vk[s]-V0[s]))
            if delta < treshold*(1-self.gamma)/self.gamma and sum([(V0[s]-Value_function[s])**2 for s in self.States]) < treshold:
                break
        return {i:{j:(1 if pi[i]== j else 0) for j in self.all_info[i]} for i in self.all_info}
      
#Test
P:Transitions_Rewards_Action_A={
    'Food':{
        'a':({'Food':0.4, 'Game':0.6},-2),
        'b':({'Food':1/3,'Sleep':1/3,'Game':1/3},-5),
        'c':({'Sleep':1},0)
    },
    'Sleep':{
        'a':({'Sleep':0.2, 'Game':0.8},2),
        'c':({'Sleep':0.7, 'Game':0.3},0)
    },
    'Game':{
        'a':({'Food':0.9, 'Sleep':0.1},-1),
        'b':({'Food':1/3,'Sleep':1/3,'Game':1/3},2)
    }
}
Pol:Policy={
    'Food':{'a':0.5,'b':0.25,'c':0.25},
    'Sleep':{'a':0.8,'c':0.2},
    'Game':{'a':0,'b':1}
}
