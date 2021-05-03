import math
import pickle

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from numba import cuda
from numba import njit, prange

cuda.select_device(0)


@cuda.jit
def pdfGPU(mu, sig, X):
    """
    Probability Density function for the Gaussian Normal Distribution

    :param mu: mean of distribution
    :param sig: standard div of distribution
    :param X: point on distribution
    :return: instantaneous density of X under distribution
    """
    i = cuda.grid(1)
    j = i // X.shape[1]
    i = i % X.shape[1]

    if j < X.shape[0]:
        X[j][i] = math.exp(-math.pow(X[j][i] - mu[j], 2) / (2 * sig[j] ** 2)) / math.sqrt(2 * np.pi * sig[j] ** 2)


def cdfGPU(mu, sig, x, delta):
    """
    Simple Quad integration of the PDF between X and X+Delta

    :param mu: mean of distribution
    :param sig: standard div of distribution
    :param x:  point on distribution
    :param delta: small margin to go up the distribution (defined to make a closed area)
    :return: the probability of being this price
    """
    DX = .01

    X = np.arange(x, delta, .01).reshape((1, -1))

    threads = 32
    blocks = (X.shape[0] * X.shape[1] + threads - 1) // threads

    pdfGPU[blocks, threads](np.array([mu]), np.array([sig]), X)

    return X.sum() * DX


# @cuda.jit(device=True)
@njit()
def call_device(strike, price, curve):
    """
    Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div

    :param strike: strike price of option
    :param price: single float asset prices under distribution
    :param curve: single % chance of asset price
    :return: estimate of call price
    """

    return max(0, price - strike) * curve


# @cuda.jit(device=True)
@njit()
def put_device(strike, price, curve):
    """
    Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div

    :param strike: strike price of option
    :param price: array of asset prices under distribution
    :param curve: array of % chances of asset price
    :return: estimate of put price
    """

    return max(0, strike - price) * curve


@njit()
def call_price(strike, prices, curve):
    """
    Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div

    :param strike: strike price of option
    :param prices: array of asset prices under distribution
    :param curve: array of % chances of asset price
    :return: estimate of call price
    """

    return (np.maximum(0, prices - strike) * curve).sum()


@njit()
def put_price(strike, prices, curve):
    """
    Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div

    :param strike: strike price of option
    :param prices: array of asset prices under distribution
    :param curve: array of % chances of asset price
    :return: estimate of put price
    """

    return (np.maximum(0, strike - prices) * curve).sum()


@njit(parallel=True)
def error(X0, f_statics, price_arra, dist_arra, errn):
    """
    Given price curves, calculate the error for each option

    :param X0: (?, 3) array of the form (weight, scale, shape) for lognormal dist
    :param f_statics: array of options of form (strike_price, is_call, mark_price, error_weighting_factor)
    :param price_arra: array of asset prices under distribution
    :param dist_arra:  array of % chances of asset price for the above
    :param errn: empty array to populate with values
    :return: mutated error array
    """

    for i in prange(f_statics.shape[0]):
        for j in prange(X0.shape[0]):
            #    pij = cuda.grid(1)

            #                if p>price_arra.shape[1]:
            #                    return
            strike, typ, mark, vol = f_statics[i]
            weight, _, _ = X0[j][0], X0[j][1], X0[j][2]
            prices, curve = price_arra[j], dist_arra[j]
            if typ:
                errn[i] += call_price(strike, prices, curve) * weight
            else:
                errn[i] += put_price(strike, prices, curve) * weight


def get_price_curve(X0):
    """
    instantiate a CUDA kernel with stock prices following the distributions and with current stock price

    :param X0: (?, 3) array of form (weight, scale [Mean], shape [deviance]
    :return: price array with CDF evaluated % likelihood curve
    """
    partial = np.zeros(X0.shape[0])

    for i in range(X0.shape[0]):
        _, mu, sig = X0[i][0], X0[i][1], X0[i][2]
        partial[i] = cdfGPU(mu, sig, -500, 0)
    prices = np.stack([np.linspace(max(0, m - 7 * s), max(10, m + 7 * s, 3 * m), 100_000) for w, m, s in X0])
    curve = cuda.to_device(np.ascontiguousarray(prices.copy()))
    means = cuda.to_device(np.ascontiguousarray(X0[:, 1].tolist()))
    stds = cuda.to_device(np.ascontiguousarray(X0[:, 2].tolist()))

    threads = 32
    blocks = (curve.shape[0] * curve.shape[1] + threads - 1) // threads

    pdfGPU[blocks, threads](means, stds, curve)
    curveprime = curve.copy_to_host()
    curveprime = (curveprime.T * (prices[:, 1] - prices[:, 0]) / (1 - partial)).T

    return prices, curveprime


def bayes_error(X0: np.array, f_statics, curStock) -> int:
    """
    Calculate the Root Sum Squared Weighted Error weighted by the vol column in f_statics

    :param X0: (?, 3) array of form (weight, mean, std)
    :param f_statics: (?, 4) array of form (strike_price, is_call, mark_price, error_weighting_factor)
    :param curStock: the current stock price of the underlying asset
    :return: Root Sum Squared Weighted Error
    """
    prices, curve = get_price_curve(X0)
    errn = np.zeros(f_statics.shape[0])

    # threads = 32
    # blocks = (curve.shape[0] * curve.shape[1] + threads - 1) // threads
    error(X0, f_statics, prices, curve, errn)

    errn = np.sqrt((((errn - f_statics[:, 2]) ** 2 * f_statics[:, 3]) / f_statics[:, 3].sum()).sum())

    return errn


def static_array(f_statics, ticker=b'GME') -> np.array:
    """
    Transform the f_statics dictionary into a numba conformal array

    :param f_statics: the f_statics dictionary loaded from the message pack file
    :param ticker: the specific ticker to extract from the f_statics dictionary
    :return: np.array containing only floats so it may be passed into Numba decorated functions
    """
    return np.array([[strike,
                      1.0 if y == b'C' else 0.0,
                      float(data.get(b'mark_price')),
                      np.log(float(data.get(b'open_interest')) + 1)]
                     for strike, opt in f_statics[ticker].items()
                     for y, data in opt.items()])


def PDF(x0, x_spread):
    """
    Calculate the Values for the probability Density function across the SPREAD array initiated at top of file

    :param x0: The minimization result array
    :param x_spread: the X values to evaluate the PDF at
    :return: The PDF calculated for each element in the spread array
    """
    x0 = x0.reshape((-1, 3))
    # res2 = np.zeros_like(x_spread)
    xx = np.stack([x_spread for _ in range(x0.shape[0])])
    xx = np.ascontiguousarray(xx[:].tolist())
    means = np.ascontiguousarray(x0[:, 1].tolist())
    stds = np.ascontiguousarray(x0[:, 2].tolist())

    threads = 32
    blocks = (xx.shape[0] * xx.shape[1] + threads - 1) // threads
    pdfGPU[blocks, threads](means, stds, xx)

    xx = (xx.T * x0[:, 0]).sum(1)
    # dists = [(w, scipy.stats.norm(m, s)) for w, m, s in X0]
    # for i in range(x_spread.shape[0]):
    #     res2[i] = sum(w * d.pdf(x_spread[i]) for w, d in dists)

    return xx / x0[:, 0].sum(),  # res2


def CDF(x0, x_spread):
    """
    Calculate the Values for the Cumulative density function across the SPREAD array initiated at the top of file

    :param x0: the minimization result array
    :param x_spread: the X values to evaluate the CDF at
    :return: The CDF calculated for each element in the x_spread array
    """
    x0 = x0.reshape((-1, 3))
    dists = [(w, scipy.stats.norm(m, s), scipy.stats.norm(m, s).cdf(0)) for w, m, s in x0]
    return np.array([sum(w * (d.cdf(x) - partial) / (1 - partial) for w, d, partial in dists) for x in x_spread])


def CI(X0: np.array, f_static: np.array):
    """
    Calculate the 5% and 95% confidence interval

    :param X0: the minimization result array
    :param f_static: the statics
    :return: the 2 confidence intervals
    """
    ub = max(f_static[:, 0]) / 2
    dx = ub / 5
    x_spread = np.linspace(0, ub, 1001)
    cd = CDF(X0, x_spread)
    while (cd < .05).all():
        x_spread = np.linspace(ub, ub + dx, 201)
        ub += dx
        cd = CDF(X0, x_spread)
    five = x_spread[np.array(CDF(X0, x_spread)) < .05][-1]
    while not (cd > .95).any():
        x_spread = np.linspace(ub, ub + dx, 201)
        ub += dx
        cd = CDF(X0, x_spread)
    ninefive = x_spread[np.array(CDF(X0, x_spread)) > .95][0]

    return five, ninefive


def graphs(m, f_statics, x_spread, curStock):
    """
    Plot out the 5 different Charts to visualize the minimization

    :param m: the minimization
    :param f_statics:  the f_statics to compare against
    :param x_spread: the X values to evaluate the curves at
    :param curStock: the current stock price of the underlying asset
    :return: Nothing
    """

    price, curve = get_price_curve(np.ascontiguousarray(m.x))

    plt.annotate("Profitable", (0, 10))
    plt.annotate("OverPriced", (0, -10))
    plt.hlines(0, f_statics[:, 0].min(), f_statics[:, 0].max())

    plt.scatter([k for k, pc, _, _ in f_statics if pc],
                [sum(call_price(k, price[j], curve[j]) * m.x[j][0] for j in range(m.x.shape[0])) - p
                 for k, pc, p, _ in f_statics if pc],
                c='red')
    plt.scatter([k for k, pc, *_ in f_statics if not pc],
                [sum(put_price(k, price[j], curve[j]) * m.x[j][0] for j in range(m.x.shape[0])) - p
                 for k, pc, p, _ in f_statics if not pc],
                c='blue')
    plt.legend(["breakeven", "call", "put"])
    plt.title("Price Differences")
    plt.figure()
    plt.title("Calls")
    plt.scatter([k for k, pc, _, _ in f_statics if pc],
                [sum(call_price(k, price[j], curve[j]) * m.x[j][0] for j in range(m.x.shape[0]))
                 for k, pc, p, _ in f_statics if pc],
                c='red')
    plt.scatter([k for k, pc, *_ in f_statics if pc],
                [p
                 for k, pc, p, _ in f_statics if pc],
                c='blue')
    plt.legend(['Estimated', 'Actual Price'])
    print(*m.x.reshape((-1, 3)).tolist(), sep='\n')

    plt.figure()
    plt.scatter(f_statics[:, 0], f_statics[:, -1])
    plt.title("Volume")

    plt.figure()
    for i in range(1, m.x.shape[0] + 1):
        pfds = PDF(m.x[i - 1:i], x_spread)
        plt.plot(x_spread, -pfds[0])
    plt.plot(x_spread, PDF(m.x, x_spread)[0], lw=5)
    plt.ylim(-.1, .1)

    plt.figure()
    for i in range(1, m.x.shape[0] + 1):
        plt.plot(x_spread, CDF(m.x[i - 1:i], x_spread), )
    plt.plot(x_spread, CDF(m.x, x_spread), lw=5)
    l, h = CI(m.x, f_statics)
    print(l, h)
    plt.vlines([l, h], 0, 1, colors='k')
    plt.show(block=True)


def analyze(f_statics, cur_stock):
    """
    Load up the minimization and list of intermediate minimizations from file and run analysis on them

    :param f_statics: the f_statics to run against
    :param cur_stock the current stock price of the underlying asset
    :return: Nothing
    """

    # Used for the display after optimizing - NOT Used in optimization
    x_spread = np.linspace(0.01,
                           max(f_statics[:, 0]),
                           1000)

    with open('pick.le', 'rb') as fin:
        m = pickle.load(fin)
    # with open('dill.pickle', 'rb') as fin:
    #     ma = pickle.load(fin)
    print(m)
    graphs(m, f_statics, x_spread=x_spread, curStock=cur_stock)
    # for mx in ma:
    #     m.x = mx
    #     graphs(m, f_statics)
