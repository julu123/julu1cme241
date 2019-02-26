from MDP_A import MDP_A
from MDP_B import MDP_B
from Tabular_RL_Algorithms import PredictionMethods
from Options import Option
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



op_pol = MDP_B(P).get_optimal_policy()
#print(op_pol)
#print(MDP_B(P).get_optimal_value_function())

print(MDP_B(P).policy_evaluation(pol))
random.seed(1)
print(PredictionMethods(P, pol).td_zero())
random.seed(1)
print(PredictionMethods(P, pol).td_lambda(method="Backward"))
random.seed(1)
print(PredictionMethods(P, pol).monte_carlo_first_visit())



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

