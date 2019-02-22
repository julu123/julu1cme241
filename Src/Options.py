import numpy as np
from scipy.stats import norm
from scipy.stats import linregress

class Option:
    def __init__(self, sigma: float, maturity: float, rf: float):
        self.sigma = sigma
        self.rf = rf
        self.maturity = maturity

    def black_scholes_price(self, stock_price: float,
                            strike: float,
                            call_or_put: str = "Call"):
        d1_num = np.log(stock_price / strike) + (self.rf + self.sigma ** 2 / 2) * self.maturity
        d1 = d1_num / (self.sigma * np.sqrt(self.maturity))
        d2 = d1 - self.sigma * np.sqrt(self.maturity)
        if call_or_put == "Call":
            return norm.cdf(d1)*stock_price - norm.cdf(d2)*strike*np.exp(-self.rf*self.maturity)
        elif call_or_put == "Put":
            return norm.cdf(-d2)*strike*np.exp(-self.rf*self.maturity)-norm.cdf(-d1)*stock_price

    def binomial_tree_price(self,
                            stock_price: float,
                            strike: float,
                            call_or_put: str = "Call",
                            origin: str = "European",
                            n: int = 200):
        dt = self.maturity/n
        u = np.exp(self.sigma*np.sqrt(dt))
        d = 1/u
        p = (np.exp(self.rf*dt) - d)/(u-d)
        stock_matrix = np.zeros((n, n))
        price_matrix = np.zeros((n, n))
        stock_matrix[0, 0] = stock_price
        for i in range(1,n):
            for j in range(i):
                stock_matrix[j,i] =stock_matrix[j,i-1]*u
            stock_matrix[i,i] = stock_matrix[i-1,i-1]*d
        for i in range(n):
            if call_or_put == "Call":
                price_matrix[i,n-1] = max(stock_matrix[i, n-1]-strike, 0)
            elif call_or_put == "Put":
                price_matrix[i, n - 1] = max(strike - stock_matrix[i, n - 1], 0)
        for i in reversed(range(n-1)):
            for j in range(i+1):
                if origin == "American" and call_or_put == "Put":
                    price_matrix[j,i] = max(np.exp(-self.rf*dt) * (
                            p*price_matrix[j, i+1] + (1-p)*price_matrix[j+1, i+1]),
                                            strike-stock_matrix[j, i]
                                            )
                else:
                    price_matrix[j, i] = np.exp(-self.rf * dt) * (
                                p * price_matrix[j, i + 1] + (1 - p) * price_matrix[j + 1, i + 1])
        return price_matrix[0,0]

    def longstaff_schartz_price(self,
                          stock_price: float,
                          strike: float,
                          m: int = 10000,
                          n: int = 200): #call_or_put is an american put (no need to implement)
        "m is the amount of simulations, n is the amount of time steps"
        dt = self.maturity/n
        normal_returns = norm.rvs(size=(m, 1))
        final_returns = (self.rf - self.sigma**2/2)*self.maturity + self.sigma*np.sqrt(self.maturity)*normal_returns
        final_stock_prices = stock_price*np.exp(final_returns)
        final_payoffs = np.maximum(strike-final_stock_prices, 0)
        for i in reversed(range(1, n)):
            normal_returns = norm.rvs(size=(m, 1))
            returns = final_returns*i/(i+1) + self.sigma*np.sqrt(i/(i+1))*normal_returns
            stock_prices = stock_price*np.exp(returns)

            in_the_money_stocks = stock_prices[stock_prices < strike]
            discounted_payoffs = np.exp(-dt*self.rf) * final_payoffs[stock_prices < strike]

            slope, intercept, _, _, _ = linregress(in_the_money_stocks, discounted_payoffs)

            estimated_continued_value = slope * stock_prices
            exercise_values = np.maximum(strike - stock_prices, 0)

            final_payoffs[estimated_continued_value > exercise_values] = \
                final_payoffs[estimated_continued_value > exercise_values]*np.exp(-self.rf*dt)
            final_payoffs[estimated_continued_value <= exercise_values] = \
                exercise_values[estimated_continued_value <= exercise_values]

        return np.mean(final_payoffs*np.exp(-self.rf*dt))
