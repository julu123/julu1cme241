from Algorithms.LSPI_options import LPSI
from Algorithms.Options import Option
#from Algorithms.ApproximatePredictionAlgorithms import MonteCarlo


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

param = {'mu': 0.1, 'sigma': 0.2}

features = {'stock price': 100, 'ttm': 2, 'strike': 110}

rf = 0.005

#print(Option(param['sigma'], 2, rf).binomial_tree_price(stock_price=features['stock price'],
#                                                        strike=features['strike'],
#                                                        call_or_put="Put",
#                                                        origin="American"))

#for i,j in enumerate(pol):
#    print(i, j)

#print(len(features))

print(LPSI().learn_2())
#print(LPSI().learn_test())
print(LPSI().learn())
print(Option().binomial_tree_price())