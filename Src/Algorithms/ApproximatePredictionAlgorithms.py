import numpy as np
from Algorithms.FunctionApproximationBase import FunctionApproximationBase
from Utils.GBMStockSimulator import GMBStockSimulator

# This is very much a WIP
# Model assumes that price of option = a * ttm + b * stock_price + c * exercise_value


class MonteCarlo(FunctionApproximationBase):
    def __init__(self,
                 alpha: float = 0.1,
                 strike: float = 110,
                 features: dict = {'stock price': 100, 'ttm': 2, 'payoff': 0}
                 ):
        self.strike = strike
        self.alpha = alpha
        self.features = features

    def initialize_theta(self):
        return np.random.rand(len(self.features), 1) * 0.01

    def update_theta(self,
                     theta,
                     observed_gt: float,
                     estimated_value,
                     gradient):
        # theta and gradient are both vectors of the same size, all the other values are scalar
        return theta + self.alpha * (observed_gt - estimated_value) * gradient

    def update_features(self, features, price, step_size):
        return {'stock price': price, 'ttm': (features['ttm']-step_size), 'payoff': max(0, price - self.strike)}

    def get_feature_vector(self, features):
        vector = np.zeros((1, len(features)))
        for i, j in enumerate(features):
            vector[0, i] = features[j]
        return vector

    def learn(self,
              episode_size: int = 50,
              nr_episodes: int = 100):
        theta = self.initialize_theta()
        step_size = self.features['ttm']/episode_size
        for i in range(nr_episodes):
            features = self.features
            stock_data = StockSimulator(step_size).generate_n_step_training_data(stock_price=features['stock price'],
                                                                                 time=features['ttm'])
            for j in range(episode_size):
                features = self.update_features(features, stock_data[j], step_size)
                if j < episode_size:
                    g_t = 0
                else:
                    g_t = max(0, stock_data[j] - self.strike)
                state_features = self.get_feature_vector(features)
                gradient = state_features.T
                e_v = np.dot(state_features, theta)
                if j % 10 == 0:
                    print(theta)
                theta = self.update_theta(theta, g_t, e_v, gradient)
        return theta
