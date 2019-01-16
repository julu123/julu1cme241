#!/usr/bin/env python
# coding: utf-8

# This code returns an accumulated reward as well (does not use the reward from the starting point though
# This code is basically the same as the MP, just with and added reward
def MarkovReward(changes,P,states,reward,start,discount,print_text=False):
    import numpy as np
    import random
    # Changes = amount of times the process will change (integer)
    # P = transition matrix (matrix)
    # states = possible states (list)
    # reward = reward at each state (list of numerics values)
    # start = start value in transition matrix (string)
    # discount = discount factor between each state (numeric)
    accumulated_reward = 0
    path=[start]
    current_activity = start
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
                        accumulated_reward += reward[k]
                        if i > 0:
                            accumulated_reward = accumulated_reward*discount
                        #print("k:", k)
                        break
                break
        i += 1
    if print_text == True:
        print("The path was:", path)
        print("The procces resulted in a reward of: ", accumulated_reward, ".")
    return(path,accumulated_reward)


#My MRP

states=[["School", "Game", "Food", "Party"],[5, 3, 1, -4]]
P=[[0.5,0.2,0.2,0.1],[0,1,0,0],[0.3,0.1,0.3,0.3],[0,0.1,0.6,0.3]]


q,z=MarkovReward(50,P,states[0],states[1],states[0][0],1,print_text=True)

# It will return something like:
The procces reached the termination state ' Game ' after 7 steps.
The path was: ['School', 'Food', 'Party', 'Food', 'School', 'School', 'School', 'Game']
The procces resulted in a reward of:  16 .




