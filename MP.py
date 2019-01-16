#!/usr/bin/env python
# coding: utf-8

# This is a very basic code for simulating a Markov Process. It returns the path the Process takes.

def MarkovSimulation(changes,P,states,start,print_text=False):
    import numpy as np
    import random
    # Changes = amount of times the process will change (integer)
    # P = transition matrix (matrix)
    # states = possible states (list)
    # start = start value in transition matrix (string)
    i = 0
    path=[start]
    current_activity = start
    while i < changes:
        for j in range(len(P)):
            if current_activity == states[j]:
                #print("j:", j ) Turn on for debugging. Every change in k has to be followed by the same change in j! 
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
                        #print("k:", k) debugging
                        break
                break
        i += 1
    if print_text == True:
        print("The path was:", path)
    return(path)

# These are my values. Transitions is a list that labels every possible transition, 
# e.g. Game to Game is labeled GG and has a probability of zero.

states=["School", "Game", "Food", "Party"]
P=[[0.5,0.2,0.3,0],[0,0,0.5,0.5],[0.3,0.3,0.1,0.3],[0,0.2,0.5,0.3]]



#Simulate 7 steps, starting at "School" (States[0])
q=MarkovSimulation(7,P,states,states[0])






