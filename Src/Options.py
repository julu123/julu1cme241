import numpy as np
from scipy.stats import norm

class European_option:
    def __init__(self, sigma: float, maturity: float, rf: float):
        self.sigma = sigma
        self.rf = rf
        self.maturity = maturity

    def black_scholes_price(self, stock_price: float, strike: float, type: str = "Call"):
        d1_num = np.log(stock_price / strike) + (self.rf + self.sigma ** 2 / 2) * self.maturity
        d1 = d1_num / (self.sigma * np.sqrt(self.maturity))
        d2 = d1 - self.sigma * np.sqrt(self.maturity)
        if type == "Call":
            return norm.cdf(d1)*stock_price - norm.cdf(d2)*strike*np.exp(-self.rf*self.maturity)
        elif type == "Put":
            return norm.cdf(-d2)*strike*np.exp(-self.rf*self.maturity)-norm.cdf(-d1)*stock_price

    def binomial_tree_price(self, stock_price: float, strike: float, type: str = "Call", n: int = 200):
        t = self.maturity/n
        u = np.exp(self.sigma*np.sqrt(t))
        d = 1/u
        p = (np.exp(self.rf*t) - d)/(u-d)
        stock_matrix = np.zeros((n, n))
        price_matrix = np.zeros((n, n))
        stock_matrix[0, 0] = stock_price
        for i in range(1,n):
            for j in range(i):
                stock_matrix[j,i] =stock_matrix[j,i-1]*u
            stock_matrix[i,i] = stock_matrix[i-1,i-1]*d
        for i in range(n):
            if type == "Call":
                price_matrix[i,n-1] = max(stock_matrix[i, n-1]-strike, 0)
            elif type == "Put":
                price_matrix[i, n - 1] = max(strike - stock_matrix[i, n - 1], 0)
        for i in reversed(range(n-1)):
            for j in range(i+1):
                price_matrix[j,i] = np.exp(-self.rf*t)*(p*price_matrix[j, i+1] + (1-p)*price_matrix[j+1, i+1])
        return price_matrix[0,0]

    def longstaff_schartz(self, stock_price: float, strike:float, m:int = 500, n:int = 200): #Type is an american put (no need to implement)
        pass