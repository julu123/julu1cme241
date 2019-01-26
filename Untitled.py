#!/usr/bin/env python
# coding: utf-8

import numpy as np
from typing import TypeVar, Dict, List, Tuple

Action = TypeVar('Action')
Transitions_Rewards_Action_A=Dict[State,Dict[Action,Dict[Tuple[State,(float or int)],(float or int)]]]
Policy = Dict[State,Tuple[Action,(float or int)]]

class MDP_A(MRP_A):
    def __init__(self,ProbDist:Transitions_Rewards_Action_A=None,gamma:float=1):
        self.States=list(ProbDist)
        self.gamma=gamma
        self.all_info=ProbDist
        pass
    
    def Get_MRP(self,Pol:Policy):
        ProbDist=np.zeros((len(self.States),len(self.States)))
        rew=np.zeros((len(self.States),1))
        for i in self.all_info:
            for j in self.all_info[i]:
                rew[list(self.all_info).index(i)] += self.all_info[i][j][1]*Pol[i][j]
                for k in self.all_info[i][j][0]:
                    ProbDist[list(self.all_info).index(i)][list(self.all_info).index(k)] += (self.all_info[i][j][0][k]*Pol[i][j])
        rewards = [float(i) for i in rew]
        return(MRP_A(ProbDist,self.gamma,rewards,self.States))
    
    def Get_Optimal_Policy(self):
        pass
      
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
Pol:Policies={
    'Food':{'a':0.5,'b':0.25,'c':0.25},
    'Sleep':{'a':0.8,'c':0.2},
    'Game':{'a':0,'b':1}
}
