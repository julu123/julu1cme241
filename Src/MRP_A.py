#!/usr/bin/env python
# coding: utf-8

# In[776]:


import numpy as np
from typing import TypeVar, Dict, List, Tuple
from MP import MP
from Variables import State, States, Transitions, Transitions_rewards, R_A


class MRP_A(MP):
    def __init__(self,ProbDist:(Transitions_rewards or np.ndarray),
                 gamma:float=1,
                 R:R_A=None,
                 S:States=None,
                 print_text=False):
        if S == None:
            if isinstance(ProbDist, np.ndarray) == False:
                self.States=list(ProbDist)
            else:
                self.States=[i for i in range(len(ProbDist))]
        else:
            self.States=S
        if np.all(R) == None:
            Rew=[]
            for i in ProbDist:
                for j in ProbDist[i]:
                    if isinstance(j, (int,float)) == True:
                        Rew.append(j)
            self.Rewards = Rew
        else:
            assert len(R) == len(self.States)
            self.Rewards = R
        if isinstance(ProbDist, np.ndarray) == True:
            self.ProbDist = ProbDist
        else:
            self.ProbDist=self.Get_ProbDist(ProbDist)
        self.gamma=gamma
        self.print_text=print_text
        
    def Get_Value_Function(self,n:int=None):
        if np.linalg.det(self.ProbDist) > 1e-5:
            R=np.dot(self.ProbDist,self.Rewards)
            inverse=np.linalg.inv(np.identity(len(self.States))-self.gamma*self.ProbDist)
            return(list(np.dot(inverse,R)))
        else:
            print("Determinant Zero -- wait for simulation")
            VF=np.zeros((len(self.States),1))
            for item in self.States:
                if n == None:
                    n=1000
                test=0
                for i in range(n):
                    test+=self.Simulate_Rewards(n,item)
                VF[self.States.index(item)]=test/n
            return(VF)
    
    def Get_ProbDist(self,ProbDist):
        assert isinstance(ProbDist,dict) == True
        Trans_mat=np.zeros((len(self.States),len(self.States)))
        for i in ProbDist:
            for j in ProbDist[i][0]:
                Trans_mat[self.States.index(i)][self.States.index(j)]=ProbDist[i][0][j]
        return(Trans_mat)
        
    def Simulate_Rewards(self,steps:int=10,start:State=None,print_text=False):
        if isinstance(start,(int,float)) == True:
            start=self.States[int(start)]
        path = self.Simulate(steps,start,print_text)
        accumulated_reward=0
        i=0
        for item in path:
            path_ind=self.States.index(item)
            accumulated_reward=accumulated_reward+self.Rewards[path_ind]*self.gamma**i
            i+=1
        if print_text == True:
            print('The path Resulted in a value of ', float(accumulated_reward))
        return(float(accumulated_reward))
    
    def Look_Up_Reward(self,start:State):
        if isinstance(start, (int,float)) == True:
            return self.Rewards[start]
        else:
            return self.Rewards[self.States.index(start)]
    

#Test    
P:Transitions_rewards={
   'C1':({'C2':0.5, 'FB':0.5},-2),
   'C2':({'C3':0.8,'Sleep':0.2},-2),
   'C3':({'Pass':0.6,'Pub':0.4},-2),
   'Pass':({'Sleep':1},10),
   'Pub':({'C1':0.2,'C2':0.4,'C3':0.4},1),
   'FB':({'C1':0.1,'FB':0.9},-1),
   'Sleep':({'Sleep':1},0)
}

