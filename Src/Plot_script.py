from Processes.MDP_B import MDP_B
from Algorithms.Tabular_RL_Algorithms import PredictionMethods
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
dp_matrix = np.zeros((101, 1))
td_forward_offline = np.zeros((101, 1))
td_forward_online = np.zeros((101, 1))
td_backward = np.zeros((101, 1))
axis = np.linspace(0, 1, 101)
n=500

for i in range(101):
    lambd = i/100
    vf_f = PredictionMethods(P, pol).td_lambda(lambd=lambd, method="Forward", update="Online", nr_episodes=n,
                                             episode_size=500)
    vf_b = PredictionMethods(P, pol).td_lambda(lambd=lambd, method="Backward", update="Online", nr_episodes=n,
                                             episode_size=500)
    vf_offline = PredictionMethods(P, pol).td_lambda(lambd=lambd, method="Backward", update="Online", nr_episodes=n,
                                             episode_size=500)
    td_forward_online[i] = vf_f['Sleep']
    td_forward_offline[i] = vf_offline['Sleep']
    td_backward[i] = vf_b['Sleep']
    dp_matrix[i] = dp_value['Sleep']


plt.plot(axis, dp_matrix, label="Value iteration")
plt.plot(axis, td_forward_online, label="TD-Forward (Online)")
plt.plot(axis, td_forward_offline, label="TD-Forward (Offline)")
plt.plot(axis, td_backward, label="TD-Backward")
plt.legend()
plt.ylabel('Value function of State "Sleep" ')
plt.xlabel('\lambda')
plt.show()
