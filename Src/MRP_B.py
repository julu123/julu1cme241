import numpy as np
from typing import TypeVar, Dict, List, Tuple
from MP import MP
from Variables import State, States, Transitions, PR, Transitions_rewards_B, R_B


class MRP_B(MP):
    def __init__(self,ProbDist:(Transitions_rewards_B or np.ndarray),
                 gamma:float=1,
                 R:(R_B or np.ndarray)=None,
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
            self.Rewards=self.Get_Reward_matrix(ProbDist)
        else:
            assert len(R) == len(self.States)
            self.Rewards=R   
        if isinstance(ProbDist, np.ndarray) == True:
            self.ProbDist = ProbDist
        else:
            self.ProbDist=self.Get_ProbDist(ProbDist)
        self.gamma=gamma
        self.print_text=print_text
    
    def Get_ProbDist(self,ProbDist):
        assert isinstance(ProbDist,dict) == True
        Trans_mat=np.zeros((len(self.States),len(self.States)))
        for i in ProbDist:
            for j in ProbDist[i]:
                Trans_mat[self.States.index(i)][self.States.index(j)]=ProbDist[i][j][0]
        return(Trans_mat)
    
    def Get_Reward_matrix(self,ProbDist):
        assert isinstance(ProbDist,dict) == True
        Trans_mat=np.zeros((len(self.States),len(self.States)))
        for i in ProbDist:
            for j in ProbDist[i]:
                Trans_mat[self.States.index(i)][self.States.index(j)]=ProbDist[i][j][1]
        return(Trans_mat)
    
    def Convert_to_A(self):
        rewards=np.sum((np.multiply(self.ProbDist,self.Rewards)),axis=1,keepdims=True)
        rew=[float(i) for i in rewards]
        return(MRP_A(self.ProbDist,self.gamma,rew,self.States))

       
#Test    
P:Transitions_rewards={
   'C1':{'C2':(0.5,-2), 'FB':(0.5,-0.2)},
   'C2':{'C3':(0.8,-1),'Sleep':(0.2,-5)},
   'C3':{'Pass':(0.6,10/0.6),'Pub':(0.4,1/0.4)},
   'Pass':{'Sleep':(1,1)},
   'Pub':{'C1':(0.2,-7),'C2':(0.4,-2.5),'C3':(0.4,-3)},
   'FB':{'C1':(0.1,-6),'FB':(0.9,-1)},
   'Sleep':{'Sleep':(1,0)}
}

