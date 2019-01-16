#!/usr/bin/env python
# coding: utf-8

# In[116]:


def MarkovReward(changes,P,states,start,discount,reward,print_text=False):
    import numpy as np
    import random
    # Changes = amount of times the process will change (integer)
    # P = transition matrix (matrix)
    # states = possible states (list)
    # reward = reward at each state (list of numerics values)
    # start = start value in transition matrix (string)
    # discount = discount factor between each state (numeric)
    path=[start]
    current_activity = start
    if reward != None:
        accumulated_reward = 0   
    if discount == None:
        discount = 1
    i = 0
    while i < changes:
        for j in range(len(P)):
            if current_activity == states[j]:
                #print("j:", j )
                RV = np.random.choice(states,replace=True,p=P[j])
                for k in range(len(P)):
                    if RV == states[k]:
                        if P[j][k] == 1 and k == j:
                            if print_text == True:
                                print("The procces reached the termination state", "'",states[j],"'", "after", i, "steps.")
                            i = changes
                            break
                        path.append(states[k])
                        current_activity = states[k]
                        if reward != None:
                            accumulated_reward += reward[k]
                            if i > 0:
                                accumulated_reward = accumulated_reward*discount
                        #print("k:", k)
                        break
                break
        i += 1
    if print_text == True:
        print("The path was:", path)
        if reward != None:
            print("The procces resulted in a reward of: ", accumulated_reward, ".")
    if reward != None:
        return(path,accumulated_reward)
    else:
        return(path)


# In[144]:


states=[["School", "Game", "Food", "Party"],[5, 3, 1, -4]]
P=[[0.5,0.1,0.2,0.2],[0,1,0,0],[0.3,0.1,0.3,0.3],[0,0.1,0.6,0.3]]


# In[145]:


q=MarkovReward(20,P,states[0],states[0][0],1,states[1],print_text=True)


# In[111]:


n = 5000
EV = 0
for i in range(n):
    q,z = MarkovReward(10,P,states[0],states[0][0],1,states[1],print_text=False)
    EV += z
print(EV/n)

