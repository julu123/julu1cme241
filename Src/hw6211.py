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
        self.info = {state: {action: {state2: (self.probs[state+action, state2], price*(state + action-state2) -
                                            cost * action if action == 0 else price*(state + action
                                                                                   -state2)-cost*action - delivery)
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
