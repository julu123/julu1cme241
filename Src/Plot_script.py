from MDP_B import MDP_B
from Tabular_RL_Algorithms import PredictionMethods
import matplotlib.pyplot as plt
import numpy as np

P = {
    'Food': {
        'a': {'Food': (0.4, 1), 'Game': (0.6, 2)},
        'b': {'Food': (1/3, 3), 'Sleep': (1/3, 4), 'Game': (1/3, 3)},
        'c': {'Sleep': (1, 1)}
    },
    'Sleep': {
        'a': {'Food': (0.4, 2), 'Sleep': (0.5, 3), 'Game': (0.1, 5)},
        'c': {'Sleep': (0.7, 2), 'Game': (0.3, -3)}
    },
    'Game': {
        'a': {'Food': (0.9, -1), 'Sleep': (0.1, -2)},
        'b': {'Food': (0, -3), 'Sleep': (0, -4), 'Game': (1, 0)}
    }
}
pol = {
    'Food': {'a': 0.5, 'b': 0, 'c': 0.5},
    'Sleep': {'a': 0.5, 'c': 0.5},
    'Game': {'a': 0, 'b': 1}}


dp_value = MDP_B(P).policy_evaluation(pol)
test = np.zeros((101, 1))
test2 = np.zeros((101, 1))
axis = np.linspace(0, 1, 101)

for i in range(101):
    lambd = i/100
    vf = PredictionMethods(P, pol).td_lambda(lambd=lambd, method="Forward", update="Offline", nr_episodes=1000,
                                             episode_size=500)
    test[i] = vf['Sleep']
    test2[i] = dp_value['Sleep']
    #test[[2, i]] = vf['Sleep']
    #test[[3, i]] = vf['Game']


plt.plot(axis, test)
plt.plot(axis, test2)
plt.ylabel('Value function of "Food" ')
plt.xlabel('\lambda')
plt.show()