#!/usr/bin/env python
# coding: utf-8

# This code returns an accumulated reward as well (does not use the reward from the starting point though
# This code is basically the same as the MP, just with and added reward

def MarkovReward(changes,P,states,transitions,reward,start,discount):
    import numpy as np
    import random
    # Changes = amount of times the process will change (integer)
    # P = transition matrix (matrix)
    # states = possible states (list)
    # transitions = possible transitions in matrix, e.g. frome state A to B and so forth (strings)
    # reward = reward at each state (list of numerics values)
    # start = start value in transition matrix (string)
    # discount = discount factor between each state (numeric)
    i = 0
    accumulated_reward = 0
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
                        accumulated_reward += reward[k]
                        accumulated_reward = accumulated_reward*discount
                        #print("k:", k)
                        break
                break
        i += 1
    print("The procces resulted in a reward of: ", accumulated_reward, ".")
    return(path,accumulated_reward)


#My MRP

states=[["School", "Game", "Food", "Party"],[5, 3, 1, -4]]
transitions=[["SS","SG","SF","SP"],["GS","GG","GF","GP"],["FS","FG","FF","FP"],["PS","PG","PF","PP"]]
P=[[0.5,0.2,0.3,0],[0,0,0.5,0.5],[0.3,0.3,0.1,0.3],[0,0.2,0.5,0.3]]


q,z=MarkovReward(7,P,states[0],transitions,states[1],states[0][3],0.99)

# It will return something like:
The procces resulted in a reward of:  9.69590700352197 .
q=['Party', 'Food', 'Party', 'Food', 'Party', 'Food', 'Party', 'Game']
z=9.69590700352197 .




