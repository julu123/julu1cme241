from Processes.MDP_B import MDP_B
from Algorithms.Tabular_RL_Algorithms import PredictionMethods, ControlMethods

import random
#q = MDP_B(Question61().info)
#print(q.convert_to_A().get_optimal_value_function())
#print(q.convert_to_A().get_optimal_policy())
n = 400
#print(q.get_optimal_value_function(False, n))
#print(q.get_optimal_policy(True, n))
#print(Question61().info)
#print(q.convert_to_A().all_info)
#print(len(q.all_info[4]))

##

import numpy as np
from scipy.stats import poisson

class Question61:

    def __init__(self, n: int = 5):
        self.n = n
        self.states = [i for i in range(n)]
        self.probs = self.generate_probs()
        price = 100
        cost = 50
        delivery = 100
        self.info = {state: {action: {state2:
            (self.probs[state+action, state2], price*(state + action-state2) - cost *
                                                action if action == 0 else price*(state +
                                                    action - state2)-cost*action - delivery)
                                                        for state2 in range(action+state+1)}
                                                            for action in range(n-state)}
                                                                for state in range(n)}

    def generate_probs(self, lambd: int = 2):
        probs = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(1, i+1):
                probs[i, j] = poisson.pmf(i-j, lambd)
            probs[i, 0] = 1 - sum(probs[i, :])
        return probs


##

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

it = 400
#print('iterations: ', it)
print('Policy (4,0,0,0,0)')
print(MDP_B(Question61().info).get_optimal_value_function(easy=True, n=it))
#print(MDP_B(Question61().info).get_optimal_policy(easy=True, n=it))
test_pol = {0: {0: 0, 1: 0, 2: 0, 3: 0, 4: 1}, 1: {0: 1, 1: 0, 2: 0, 3: 0}, 2: {0: 1, 1: 0, 2: 0}, 3: {0: 1, 1: 0}, 4: {0: 1}}
print('Policy (4,3,0,0,0)')
print(MDP_B(Question61().info).policy_evaluation(pol=test_pol, n=it))
#print(ControlMethods(P).sarsa(pol))

print('lll')



#op_pol = MDP_B(P).get_optimal_policy()
#print(op_pol)
#print(MDP_B(P).get_optimal_value_function())
#print('DP:')
#print(MDP_B(P).policy_evaluation(pol))
#random.seed(1)
#print('TD0:')
#print(PredictionMethods(P, pol).td_zero())
#random.seed(1)
#print('TD-lambda with Forward (Offline):')
#print(PredictionMethods(P, pol).td_lambda(lambd=1, method="Forward", update="Offline"))
#random.seed(1)
#print('TD-lambda with Forward (Online):')
#print(PredictionMethods(P, pol).td_lambda(lambd=1, method="Forward", update="Online"))
#random.seed(1)
#print('TD-lambda with Backward (Online):')
#print(PredictionMethods(P, pol).td_lambda(lambd=1, method="Backward"))
#random.seed(1)
#print('MCFV:')
#print(PredictionMethods(P, pol).monte_carlo_first_visit())

#dp_value = MDP_B(P).policy_evaluation(pol)

#test = PredictionMethods(P, pol).compare_to_dp()


#ControlMethods(P).sarsa()



#print(MDP_B(P).policy_evaluation(pol))

#print(MDP_B(P).generate_path(Pol, steps=100))


#test = SarsaGenerator(P)
#print(TabularMC(Pol).first_visit(test, 1000))

#test2=MDP_A(P)
#print(test2.policy_Evaluation(Pol))

#print(len(test.generate_path('Food', Pol)[0]))
#print(len(test.generate_path('Food', Pol)[1]))
#print(len(test.generate_path('Food', Pol)[2]))

#test = Tabular_MC(MDP_A(P), Pol)

#print(Option(0.25, 0.5, 0.05).binomial_tree_price(100, 110, "Put", "American"))

#print(Option(0.25, 0.5, 0.05).longstaff_schartz_price(100,110))

