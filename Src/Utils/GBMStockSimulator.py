import numpy as np
from scipy.stats import norm
from Processes.Variables import State, Action


class GBMStockSimulator(object):

    def __init__(self,
                 step_size: float = 1/40,
                 mu: float = 0.005,
                 sigma: float = 0.25):
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
        #assert isinstance(state, (float or int)) is True
        s = state/strike
        ttm = maturity - time
        phi0 = 1
        phi1 = np.exp(-s/2)
        phi2 = phi1 * (1 - s)
        phi3 = phi1 * (1 - 2*s + s**2/2)
        phi_t_0 = np.sin(-time*np.pi/(2*maturity)+np.pi/2)
        phi_t_1 = np.log(ttm)
        phi_t_2 = (time/maturity)**2
        #phi_n_0 = (strike - state) * ttm
        features = np.array((phi0, phi1, phi2, phi3, phi_t_0, phi_t_1, phi_t_2)).reshape(1, 7)
        return features

    def get_full_matrix(self,
                        state: State,
                        m: int = 10000,
                        n: int = 200,
                        rf: float = 0.005,
                        sigma: float = 0.25,
                        maturity: float = 5):
        step_size = maturity / n
        return_matrix = (rf - sigma**2 / 2) * step_size + sigma * np.sqrt(step_size) * norm.rvs(size=(m, n))
        stock_matrix = np.zeros((m, n + 1))
        stock_matrix[:, 0] = state
        for i in range(1, n+1):
            stock_matrix[:, i] = stock_matrix[:, int(i - 1)]*np.exp(return_matrix[:, int(i - 1)])
        return stock_matrix