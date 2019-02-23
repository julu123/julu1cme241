import numpy as np
from typing import TypeVar, Dict, List, Tuple
from Variables import State, States, Transitions


class MP(object):
    #Defined by probability distribution (ProbDist) 
    #states (S) are not required
    def __init__(self,ProbDist:(Transitions or np.ndarray),S:States=None,print_text=False):
        if S == None:
            if isinstance(ProbDist, np.ndarray) == False:
                self.States=list(ProbDist)
            else:
                self.States=[i for i in range(len(ProbDist))]
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
        assert isinstance(ProbDist,dict) == True
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
            
    def Simulate(self,steps:int=10,start:State=None,print_text=None):
        if start == None:
            start = self.States[0]
        elif isinstance(start, (int,float)) == True:
            start = self.States[int(start)]
        if print_text == None:
            print_text = self.print_text
        path = [start]
        current_activity = start
        i=0
        while i < steps:
            if self.ProbDist[self.States.index(current_activity)][self.States.index(current_activity)] == 1:
                steps=0
                if print_text == True:
                    print("The process terminated after", i, "steps, at the state", current_activity)
            else:
                RV=np.random.choice(self.States,replace=True,p=self.ProbDist[self.States.index(current_activity)])
                current_activity = RV
                path.append(current_activity)
                i += 1
        if print_text == True:
            print("The path was:", path)
        return path
    
    def Look_up(self, start: State, too: State, steps: int = 1):
        if isinstance (start, str) == True:
            i = self.States.index(start)
        elif isinstance(start, (int,float)) == True:
            i = int(start)
        if isinstance (too, str) == True:
            j = self.States.index(too)
        elif isinstance(too, (int,float)) == True:
            j = int(too)
        return np.linalg.matrix_power(self.ProbDist, steps)[i][j]
