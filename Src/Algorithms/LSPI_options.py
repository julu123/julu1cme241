import numpy as np
from Processes.Variables import State
from Utils.GBMStockSimulator import GBMStockSimulator


class LPSI(GBMStockSimulator):
    def __init__(self,
                 starting_stock_price: State = 100,
                 strike: float = 110,
                 step_size: float = 1/12,
                 maturity: float = 2,
                 gamma: float = 0.99,
                 call_or_put: str = "Call",
                 mu: float = 0.1,
                 sigma: float = 0.2,
                 rf: float = 0.005):
        GBMStockSimulator.__init__(self,
                                   step_size=step_size,
                                   mu=mu,
                                   sigma=sigma)
        self.strike = strike
        self.starting_stock_price = starting_stock_price
        self.step_size = step_size
        self.maturity = maturity
        self.call_or_put = call_or_put
        self.gamma = gamma
        self.rf = rf

    def initialize_w(self):
        size = (self.get_features(self.starting_stock_price, self.strike, 0, self.maturity)).shape
        return np.random.rand(size[1], 1)*0.01
        #return np.zeros((size[1], 1))


    def update_w(self, A, B):
        return np.dot(np.linalg.inv(A), B)

    def update_policy(self):
        pass

    def learn_test(self,
                   nr_episodes: int = 1000,
                   k: int = 7,
                   epsilon: float = 1e-4):
        w = self.initialize_w()
        for i in range(nr_episodes):
            A = np.zeros((k, k))
            B = np.zeros((k, 1))
            exercise = False
            stock_price = self.starting_stock_price
            time = 0
            current_features = self.get_features(state=stock_price,
                                                 strike=self.strike,
                                                 time=time,
                                                 maturity=self.maturity)

            while time + self.step_size <= self.maturity and exercise is False:
                if stock_price - self.strike > np.dot(current_features, w) and stock_price - self.strike > 0:
                    exercise = True
                    r = stock_price - self.strike
                elif time + self.step_size == self.maturity:
                    exercise = True
                    r = max(0, stock_price - self.strike)
                else:
                    r = 0
                time += self.step_size
                stock_price = self.generate(state=stock_price, action=False)
                next_features = self.get_features(state=stock_price,
                                                  strike=self.strike,
                                                  time=time,
                                                  maturity=self.maturity)
                A = A + np.dot(current_features.T, (current_features - self.gamma * next_features))
                B = B + r * current_features.T
                current_features = next_features
            w = w + np.dot(np.linalg.inv(A), B)
        return w, np.dot(self.get_features(state=self.starting_stock_price, strike=self.strike, time=0, maturity=self.maturity), w)

    def learn(self,
              nr_episodes: int = 10000,
              batch_size: int = 50):
        w = self.initialize_w()
        A = np.zeros((len(w), len(w)))
        B = np.zeros((len(w), 1))
        batch_ticker = 0
        for i in range(nr_episodes):
            batch_ticker += 1
            exercise = False
            stock_price = self.starting_stock_price
            time = 0
            current_features = self.get_features(state=stock_price,
                                                 strike=self.strike,
                                                 time=time,
                                                 maturity=self.maturity)
            while time + self.step_size <= self.maturity and exercise is False:
                if stock_price - self.strike > np.dot(current_features, w) and stock_price - self.strike > 0:
                    exercise = True
                    r = stock_price - self.strike
                elif time + self.step_size == self.maturity:
                    exercise = True
                    r = max(0, stock_price - self.strike)
                else:
                    r = 0
                time += self.step_size
                stock_price = self.generate(state=stock_price, action=False)
                next_features = self.get_features(state=stock_price,
                                                  strike=self.strike,
                                                  time=time,
                                                  maturity=self.maturity)
                Q = 0
                if self.call_or_put == "Call":
                    Q = max(0, stock_price - self.strike)
                elif self.call_or_put == "Put":
                    Q = max(0, self.strike - stock_price)

                P = np.zeros((1, w.shape[0]))
                if exercise is True and Q <= np.dot(next_features, w):
                    P = next_features
                R = 0
                if Q > (np.dot(P, w)):
                    R = Q

                A_part = (current_features - np.exp(-self.rf * self.step_size)*P)
                A = A + np.dot(A_part.T, current_features)

                B = B + np.exp(- self.rf * self.step_size) * R * current_features.T

                current_features = next_features
            if batch_ticker % batch_size == 0:
                w += self.update_w(A, B)
                A = np.zeros((len(w), len(w)))
                B = np.zeros((len(w), 1))
        return w, float(np.dot(self.get_features(state=self.starting_stock_price,
                                                 strike=self.strike, time=0, maturity=self.maturity), w))

    def predict(self):
        pass

