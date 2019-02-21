import numpy as np
from typing import TypeVar, Dict, List, Tuple
from MRP_B import MRP_B
from Variables import State, States, Transitions, PR, Transitions_rewards_B, R_B, Action, Policy, Transitions_Rewards_Action_B

class MDP_B(MRP_B):
    def __init__(self,ProbDist:Transitions_Rewards_Action_B=None,gamma:float=1):
        self.States=list(ProbDist)
        self.gamma=gamma
        self.all_info=ProbDist
        actions=[]
        for i in P:
            for j in P[i]:
                actions.append(j)
        self.Actions=sorted(list(set(actions)))
        self.Actiondict={i:{k for k in self.all_info[i]} for i in self.all_info}
        
    def Get_MRP(self,Pol:Policy):
        ProbDist=np.zeros((len(self.States),len(self.States)))
        rew=np.zeros((len(self.States),len(self.States)))
        for i in self.all_info:
            for j in self.all_info[i]:
                for k in self.all_info[i][j]:
                    ProbDist[list(self.all_info).index(i)][list(self.all_info).index(k)] += (self.all_info[i][j][k][0]*Pol[i][j])
                    rew[list(self.all_info).index(i)][list(self.all_info).index(k)] += (self.all_info[i][j][k][1]*Pol[i][j])
        return(MRP_B(ProbDist,self.gamma,rew,self.States))
    
    def Convert_to_A(self):
        ProbDist_A= {i:{j:({k:self.all_info[i][j][k][0] for k in self.all_info[i][j]},
                           sum(list({self.all_info[i][j][k][0]*self.all_info[i][j][k][1] for k in self.all_info[i][j]}))) for j in self.all_info[i]} for i in self.all_info}
        return(MDP_A(ProbDist_A,self.gamma))

    def Get_Optimal_Value_Function(self,treshold:float=1e-3): #This is Value the Iteration
        V0 = dict([(s,0) for s in self.States])
        while True:
            Vk = V0.copy()
            delta = 0
            for s in self.States:
                V0[s] = max([sum([self.all_info[s][a][k][1] + self.gamma*self.all_info[s][a][k][0]*Vk[k]
                                                for k in self.all_info[s][a]])
                                           for a in self.all_info[s]])
                delta = max(delta,abs(Vk[s]-V0[s]))
            if delta < treshold*(1-self.gamma)/self.gamma:
                break
        return Vk
    
    
#Test -- (Same Policy as for MDP_A.py)
P:Transitions_Rewards_Action_B={
    'Food':{
        'a':{'Food':(0.4,1), 'Game':(0.6,2)},
        'b':{'Food':(1/3,3),'Sleep':(1/3,4),'Game':(1/3,5)},
        'c':{'Sleep':(1,6)}
    },
    'Sleep':{
        'a':{'Sleep':(0.2,7), 'Game':(0.8,9)},
        'c':{'Sleep':(0.7,9), 'Game':(0.3,10)}
    },
    'Game':{
        'a':{'Food':(0.9,-1), 'Sleep':(0.1,-2)},
        'b':{'Food':(1/3,-3),'Sleep':(1/3,-4),'Game':(1/3,-5)}
    }
}
