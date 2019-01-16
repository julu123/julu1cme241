#!/usr/bin/env python
# coding: utf-8

# In[270]:


def MarkovSimulation(changes,P,transitions,start):
    import numpy as np
    import random as rm
    # Changes = amount of times the process will change (integer)
    # P = transition matrix (matrix)
    # transitions = possible transitions in matrix, e.g. frome state A to B and so forth (strings)
    # start = start value in transition matrix (string)
    i = 0
    path=[start]
    current_activity = start
    while i < changes:
        for j in range(len(P)):
            if current_activity == states[j]:
                #print("j:", j )
                RV = np.random.choice(transitions[j],replace=True,p=P[j])
                for k in range(len(P)):
                    if RV == transitions[j][k]:
                        path.append(states[k])
                        current_activity = states[k]
                        #print("k:", k)
                        break
                break
        i += 1
    print(path)
    return(path)


# In[271]:


states=["School", "Game", "Food", "Party"]
transitions=[["SS","SG","SF","SP"],["GS","GG","GF","GP"],["FS","FG","FF","FP"],["PS","PG","PF","PP"]]
P=[[0.5,0.2,0.3,0],[0,0,0.5,0.5],[0.3,0.3,0.1,0.3],[0,0.2,0.5,0.3]]


# In[272]:


q=MarkovSimulation(7,P,transitions,states[0])


# In[ ]:




