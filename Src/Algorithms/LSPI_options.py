import numpy as np
from Processes.Variables import State
from Utils.GBMStockSimulator import GBMStockSimulator


class LPSI(GBMStockSimulator):
    def __init__(self,
                 starting_stock_price: State = 100,
                 strike: float = 110,
                 step_size: float = 1/40,
                 maturity: float = 5,
                 gamma: float = 0.995,
                 call_or_put: str = "Put",
                 sigma: float = 0.25):
        GBMStockSimulator.__init__(self,
                                   step_size=step_size,
                                   mu=1-gamma,
                                   sigma=sigma)
        self.strike = strike
        self.starting_stock_price = starting_stock_price
        self.step_size = step_size
        self.maturity = maturity
        self.call_or_put = call_or_put
        self.gamma = gamma
        self.rf = 1 - gamma

    def initialize_w(self):
        size = (self.get_features(self.starting_stock_price, self.strike, 0, self.maturity)).shape
        return np.random.randn(size[1], 1)*0.01

    def update_w(self, A, B):
        return np.matmul(np.linalg.inv(A), B)

    def get_payoff(self, stock_price: float):
        payoff = 0
        if self.call_or_put == "Call":
            payoff = max(0.0, stock_price - self.strike)
        elif self.call_or_put == "Put":
            payoff = max(0.0, self.strike - stock_price)
        return payoff

    def learn_test(self,
                   nr_episodes: int = 20000,
                   nr_steps: int = 200,
                   batch_size: int = 1000,
                   r: int = 7,
                   epsilon: float = 1e-2):
        price = []
        iteration = []
        SP = self.get_full_matrix(self.starting_stock_price,
                                  n=nr_steps, m=nr_episodes, rf=self.rf,
                                  maturity=self.maturity)
        A = np.zeros((r, r))
        B = np.zeros((r, 1))
        w = np.zeros((r, 1))

        delta_t = self.maturity/nr_steps
        i = 0
        convergence = False
        while i < nr_episodes - 1 and convergence is False:
            time = 0
            for j in range(nr_steps):
                Q = self.get_payoff(SP[i, j+1])

                current_phi = self.get_features(state=SP[i, j], strike=self.strike,
                                                time=time, maturity=self.maturity)

                next_phi = self.get_features(state=SP[i, int(j+1)], strike=self.strike,
                                             time=time+delta_t, maturity=self.maturity)

                P = np.zeros((1, r))
                if j < nr_steps and Q <= np.matmul(next_phi, w):
                    P = next_phi

                R = 0
                if Q > np.matmul(P, w):
                    R = Q

                A += np.matmul(current_phi.T, np.subtract(current_phi, np.exp(- self.rf * delta_t) * P))
                B += np.exp(- self.rf * delta_t) * R * current_phi.T
                time += delta_t
            i += 1
            if (i+1) % batch_size == 0:
                next_w = np.matmul(np.linalg.inv(A), B)
                if np.linalg.norm(w-next_w) <= epsilon:
                    convergence = True
                else:
                    w = np.matmul(np.linalg.inv(A), B)
                    A = np.zeros((r, r))
                    B = np.zeros((r, 1))
                print('iteration:', i + 1, 'price: ', np.matmul(self.get_features(
                    state=self.starting_stock_price,
                    strike=self.strike,
                    time=0,
                    maturity=self.maturity), w))
                price.append(float(np.matmul(self.get_features(
                    state=self.starting_stock_price,
                    strike=self.strike,
                    time=0,
                    maturity=self.maturity), w)))
                iteration.append(i+1)
        return np.matmul(self.get_features(
                    state=self.starting_stock_price,
                    strike=self.strike,
                    time=0,
                    maturity=self.maturity), w), price, iteration

    def learn(self,
              nr_episodes: int = 100000,
              nr_steps: int = 200,
              batch_size: int = 1000,
              r: int = 8,
              epsilon: float = 1e-2):
        A = np.zeros((r, r))
        B = np.zeros((r, 1))
        w = self.initialize_w() #  np.zeros((r, 1))
        step_size = self.maturity / nr_steps
        i = 0
        convergence = False
        while i <= nr_episodes and convergence is False:
            noise = 0  # np.random.randn()*10
            stock_price = self.starting_stock_price + noise
            for j in range(nr_steps - 1):
                # Get current features
                current_features = self.get_features(state=stock_price,
                                                     strike=self.strike,
                                                     time=j * step_size,
                                                     maturity=self.maturity)
                # Get next stock price
                next_stock_price = self.generate(state=stock_price, action=False)
                # Get next features
                next_features = self.get_features(state=next_stock_price,
                                                  strike=self.strike,
                                                  time=(j+1)*step_size,
                                                  maturity=self.maturity)
                # Get Q
                Q = self.get_payoff(next_stock_price)
                # Get P
                P = np.zeros((1, r))
                if j < nr_steps - 1 and Q <= float(np.matmul(next_features, w)):
                    P = next_features
                # Get R
                R = 0
                if Q > float(np.matmul(P, w)):
                    R = Q
                # Get A
                A_PART = np.subtract(current_features, np.exp(- self.rf * step_size) * P)
                A += np.matmul(current_features.T, A_PART)
                # Get B
                B += np.exp(- self.rf * step_size) * R * current_features.T
                # Update stock price
                stock_price = next_stock_price
            if (i+1) % batch_size == 0:
                next_w = self.update_w(A, B)
                if np.linalg.norm(w - next_w) <= epsilon:
                    break
                else:
                    w = next_w
                A = np.zeros((r, r))
                B = np.zeros((r, 1))
                print('iteration:', i+1, 'price: ', np.matmul(self.get_features(
                                        state=self.starting_stock_price,
                                        strike=self.strike,
                                        time=0,
                                        maturity=self.maturity), w))
            i += 1
        return np.matmul(self.get_features(state=self.starting_stock_price,
                                           strike=self.strike,
                                           time=0,
                                           maturity=self.maturity), w)

    def learn_2(self,
                alpha: float = 0.01,
                nr_episodes: int = 10000,
                decay: bool = True,
                batch_size: int = 1000,
                epsilon: float = 1e-2):
        w = self.initialize_w()
        for i in range(nr_episodes):
            time = 0
            exercise = False
            stock_price = self.starting_stock_price
            current_features = self.get_features(state=stock_price,
                                                 strike=self.strike,
                                                 time=time,
                                                 maturity=self.maturity)
            while exercise is False and time + self.step_size <= self.maturity:
                # Find current values
                option_value = float(np.dot(current_features, w))
                intrinsic_value = self.get_payoff(stock_price)

                # Find next values
                time += self.step_size
                stock_price = self.generate(state=stock_price, action=False)
                next_features = self.get_features(state=stock_price,
                                                  strike=self.strike,
                                                  time=time,
                                                  maturity=self.maturity)

                # Decide what to do. Policy = choose max of intrinsic value or option value
                if intrinsic_value > option_value:
                    exercise = True
                    reward = intrinsic_value
                    td_error = reward - option_value
                elif time + self.step_size == self.maturity:
                    exercise = True
                    reward = self.get_payoff(stock_price)
                    td_error = reward - option_value
                else:
                    td_error = self.gamma * float(np.dot(next_features, w)) - option_value
                u = td_error * current_features.T

                # Update features
                current_features = next_features

                # Update w
                if decay is True:
                    learning_rate = alpha - alpha * i / nr_episodes
                else:
                    learning_rate = alpha
                w = w + learning_rate * u
            if (i+1) % batch_size == 0:
                print('iteration:', i+1, 'price: ', np.matmul(self.get_features(
                                        state=self.starting_stock_price,
                                        strike=self.strike,
                                        time=0,
                                        maturity=self.maturity), w))

        return w, np.dot(self.get_features(state=self.starting_stock_price, strike=self.strike,
                                           time=0, maturity=self.maturity), w)
