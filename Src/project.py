from Algorithms.Options import Option
from Algorithms.LSPI_options import LPSI
import matplotlib.pyplot as plt
import numpy as np

LSM_VS_BIN = False
if LSM_VS_BIN is True:
    n = 100
    axis = np.linspace(0.025, 2.025, n)

    LSM_price = np.zeros((n, 1))
    BIN_price = np.zeros((n, 1))

    for i in range(n):
        sigma = 2*i/n+0.025
        LSM_price[i] = Option(sigma=sigma).longstaff_schartz_price()
        BIN_price[i] = Option(sigma=sigma).binomial_tree_price()

    plt.plot(axis, BIN_price, label="Binomial price")
    plt.plot(axis, LSM_price, label="LSM price")
    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Sigma')
    plt.show()

LSM_AND_BIN_VS_LSPI = True
if LSM_AND_BIN_VS_LSPI is True:
    price = 100
    strike = 110
    sigma = 0.25

    bin_price = Option(sigma=sigma).binomial_tree_price(stock_price=price, strike=strike)
    ls_price = Option(sigma=sigma).longstaff_schartz_price(stock_price=price, strike=strike)
    a, b, c = LPSI(sigma=sigma, starting_stock_price=price, strike=strike).learn_test()

    plt.plot(c, b, label="LSPI-price")
    plt.axhline(y=bin_price, color='r', linestyle='-', label="BPM-price")
    plt.axhline(y=ls_price, color='g', linestyle='-', label="LSM-price")
    plt.legend()
    plt.ylabel('Price')
    plt.xlabel('Iterations')
    plt.show()
