#!/usr/bin/env python
# coding: utf-8

import numpy as np
from typing import TypeVar, Dict, List

State = TypeVar('S')
States = List[State]
Transitions = Dict[State,Dict[State,(int or float)]]

class MP(object):
    #Defined by probability distribution (P) 
    #states (S) are not required
    def __init__(self,ProbDist:(Transitions or np.ndarray),S:States=None,print_text=False):
        if S == None:
            if isinstance(ProbDist, np.ndarray) == False:
                self.States=list(ProbDist)
            else:
                self.States=[i for i in range(len(P))]
        else:
            self.States=S
        if isinstance(ProbDist, np.ndarray) == True:
            self.ProbDist = ProbDist
        else:
            self.ProbDist=self.Transition_matrix_to_array(ProbDist)
        self.print_text=print_text
        if self.print_text == True:
            print('You have created a new Markov process. It has ',len(self.States), 'states.')
    
    def Transition_matrix_to_array(self,ProbDist):
        assert isinstance(P,dict) == True
        Trans_mat=np.zeros((len(self.States),len(self.States)))
        for i in ProbDist:
            for j in ProbDist[i]:
                Trans_mat[self.States.index(i)][self.States.index(j)]=ProbDist[i][j]
        return(Trans_mat)
        
    def Generate_Stationary_Dist(self,choice='dict') -> Dict[State,float]:
        eigenvalues, eigenvectors = np.linalg.eig(self.ProbDist.T)
        stat_dist=np.zeros((len(self.States),1))
        for i in range(len(eigenvalues)):
            if abs(eigenvalues[i]-1) < 1e-8:
                stat_dist = stat_dist + eigenvectors[:,i].reshape(len(self.States),1)
        output_array = (stat_dist/np.sum(stat_dist)).real
        if choice == 'dict':
            return {self.States[i]:round(float(j),4) for i,j in enumerate(output_array)}
        elif choice == 'array':
            return output_array
    def Simulate(self,steps,start:State=None,print_text=None):
        if start == None:
            start = self.States[0]
        elif isinstance(start, (int,float)) == True:
            start = self.States[start]
        if print_text == None:
            print_text = self.print_text
        path = [start]
        current_activity = start
        i=0
        while i < steps:
            for j in range(len(self.States)):
                if current_activity == self.States[j]:
                    RV = np.random.choice(self.States,replace=True,p=self.ProbDist[j])
                    for k in range(len(self.States)):
                        if RV == self.States[k]:
                            if self.ProbDist[j][k] == 1 and k == j:
                                if print_text == True:
                                    print("The procces reached the termination state", "'",self.States[j],"'", "after", i, "steps.")
                                i = steps
                                break
                            path.append(self.States[k])
                            current_activity = self.States[k]
                            break
                    break
            i += 1
        if print_text == True:
            print("The path was:", path)
        return(path)
        
#Test

P={'Sleep':{'Wake up':0.7,'Eat':0.3},
   'Wake up':{'Eat':1},
   'Eat':{'Sleep':0.5,'Eat':0.5}}




