import numpy as np
from scipy.stats import norm
from Processes.Variables import State, Action


class GBMStockSimulator(object):

    def __init__(self,
                 step_size: float = 1/12,
                 mu: float = 0.1,
                 sigma: float = 0.2):
        self.step_size = step_size
        self.mu = mu
        self.sigma = sigma

    def generate_one_step_training_data(self, stock_price: float):
        return stock_price * np.exp(self.step_size * (self.mu-self.sigma**2/2) +
                                    np.sqrt(self.step_size) * self.sigma * norm.rvs())

    def generate_n_step_training_data(self,
                                      stock_price: float,
                                      time: float):
        stock_data = [stock_price]
        for i in range(int(time/self.step_size)):
            stock_data.append(self.generate_one_step_training_data(stock_price=stock_data[i]))
        return stock_data

    def generate(self, state: State, action: Action):
        if action is True:
            return state
        else:
            return self.generate_one_step_training_data(state)

    def get_features(self, state: State, strike: float, time: float, maturity: float):
        s = state/strike
        ttm = maturity - time
        phi0 = 1
        phi1 = np.exp(-s/2)
        phi2 = phi1 * (1 - s)
        phi3 = phi1 * (1 - 2*s + s**2/2)
        phi_t_0 = np.sin(-time*np.pi/(2*maturity)+np.pi/2)
        phi_t_1 = np.log(ttm)
        phi_t_2 = (time/maturity)**2
        features = np.array((phi0, phi1, phi2, phi3, phi_t_0, phi_t_1, phi_t_2)).reshape(1, 7)
        return features
