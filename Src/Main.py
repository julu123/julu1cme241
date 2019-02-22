from MDP_A import MDP_A
from Tabular_RL_Algorithms import Tabular_MC
from Options import European_option

P = {
    'Food':{
        'a':({'Food':0.4, 'Game':0.6},-2),
        'b':({'Food':1/3,'Sleep':1/3,'Game':1/3},-5),
        'c':({'Sleep':1},0)
    },
    'Sleep':{
        'a':({'Sleep':0.2, 'Game':0.8},2),
        'c':({'Sleep':0.7, 'Game':0.3},0)
    },
    'Game':{
        'a':({'Food':0.9, 'Sleep':0.1},-1),
        'b':({'Food':1/3,'Sleep':1/3,'Game':1/3},2)
    }
}
Pol = {
    'Food':{'a':0.5,'b':0.25,'c':0.25},
    'Sleep':{'a':0.8,'c':0.2},
    'Game':{'a':0,'b':1}}

#print(len(test.generate_path('Food', Pol)[0]))
#print(len(test.generate_path('Food', Pol)[1]))
#print(len(test.generate_path('Food', Pol)[2]))

#test = Tabular_MC(MDP_A(P), Pol)

print(European_option(0.25, 0.5, 0.05).binomial_tree_price(100, 110, "Put"))

print(European_option(0.25, 0.5, 0.05).longstaff_schartz(100,110))