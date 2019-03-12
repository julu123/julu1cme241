from Algorithms.Options import Option
from Algorithms.LSPI_options import LPSI
import matplotlib.pyplot as plt
import numpy as np

LSM_VS_BIN = False
if LSM_VS_BIN is True:
    n = 100
    axis = np.linspace(0, 2, n)

    LSM_price = np.zeros((n, 1))
    BIN_price = np.zeros((n, 1))

    for i in range(n):
        sigma = 2*i/n+0.01
        LSM_price[i] = Option(sigma=sigma).longstaff_schartz_price()
        BIN_price[i] = Option(sigma=sigma).binomial_tree_price()

    plt.plot(axis, BIN_price, label="Binomial price")
    plt.plot(axis, LSM_price, label="LSM price")
    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Sigma')
    plt.show()

LSM_VS_LSPI = True
if LSM_VS_LSPI is True:
    price = 100
    strike = 110
    sigma = 0.25
    #print(LPSI(sigma=sigma, starting_stock_price=price, strike=strike).learn_2(decay=False, alpha=0.01))
    print(Option(sigma=sigma).binomial_tree_price(stock_price=price, strike=strike))
    print(Option(sigma=sigma).longstaff_schartz_price(stock_price=price, strike=strike))
    print(LPSI(sigma=sigma, starting_stock_price=price, strike=strike).learn_algo31())

