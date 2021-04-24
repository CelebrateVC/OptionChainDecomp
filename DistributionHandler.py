import numpy as np
from numba import njit, prange
import matplotlib.pyplot as plt
import scipy.stats
import pickle


@njit()
def pdf(mu, sig, X):
    """
    Probability Density function for the Gaussian Normal Distribution

    :param mu: mean of distribution
    :param sig: standard div of distribution
    :param X: point on distribution
    :return: instantaneous density of X under distribution
    """
    return np.exp(-np.power(X - mu, 2) / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)


@njit()
def cdf(mu, sig, x, delta):
    """
    Simple Quad integration of the PDF between X and X+Delta

    :param mu: mean of distribution
    :param sig: standard div of distribution
    :param x:  point on distribution
    :param delta: small margin to go up the distribution (defined to make a closed area)
    :return: the probability of being this price
    """
    DX = np.linspace(x, x + delta)[1] - x
    return pdf(mu, sig, np.linspace(x, x + delta)).sum() * DX


@njit()
def call(strike, mu, sig, partial):
    """
    Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div

    :param strike: strike price of the option
    :param mu: mean of distribution
    :param sig: standard deviation of distribution
    :param partial: the Survival Function at 0 to scale the CDF by
    :return: estimate of call price
    """
    su = 0.0
    DX = np.linspace(0, max(mu * 3, 10), 500)[1]

    for i in np.linspace(0, max(mu * 3, 10), 500):
        su += max(0, i - strike) * cdf(mu, sig, i, DX) / (1 - partial)
    return su


@njit()
def put(strike, mu, sig, partial):
    """
    Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div

    :param strike: strike price of option
    :param mu: mean of distribution
    :param sig: standard deviation of distribution
    :param partial: the Survival Function at 0 to Scale the CDF by
    :return: estimate of put price
    """
    su = 0.0
    DX = np.linspace(0, max(mu * 3, 10), 500)[1]
    for i in np.linspace(0, max(mu * 3, 10), 500):
        su += max(0, strike - i) * cdf(mu, sig, i, DX) / (1 - partial)
    return su


@njit(parallel=True)
def bayes_error(X0: np.array, f_statics) -> int:
    """
    Calculate the Root Sum Squared Weighted Error weighted by the vol column in f_statics

    :param X0: (?, 3) array of form (weight, mean, std)
    :param f_statics: (?, 4) array of form (strike_price, is_call, mark_price, error_weighting_factor)
    :return: Root Sum Squared Weighted Error
    """
    partial = np.zeros(X0.shape[0])
    for i in prange(X0.shape[0]):
        weight, mu, sig = X0[i][0], X0[i][1], X0[i][2]
        for r in np.arange(-50, 0, .01):
            partial[i] += cdf(mu, sig, r, .01)
    errn = np.zeros(f_statics.shape[0])
    for i in prange(f_statics.shape[0]):
        strike, typ, mark, vol = f_statics[i]
        for j in prange(X0.shape[0]):
            weight, mu, sig = X0[j][0], X0[j][1], X0[j][2]
            part = partial[j]
            if typ:
                errn[i] += call(strike, mu, sig, part) * weight
            else:
                errn[i] += put(strike, mu, sig, part) * weight
        errn[i] -= mark
        errn[i] = errn[i] ** 2 * vol
    errn = errn / f_statics[:, 3].sum()
    return np.sqrt(errn.sum())


def static_array(f_statics, Ticker=b'GME') -> np.array:
    """
    Transform the f_statics dictionary into a numba conformal array

    :param f_statics: the f_statics dictionary loaded from the message pack file
    :param Ticker: the specific ticker to extract from the f_statics dictionary
    :return: np.array containing only floats so it may be passed into Numba decorated functions
    """
    return np.array([[strike,
                      1.0 if y == b'C' else 0.0,
                      float(data.get(b'mark_price')),
                      np.log(float(data.get(b'open_interest')) + 1)]
                     for strike, opt in f_statics[Ticker].items()
                     for y, data in opt.items()])


def PDF(X0, x_spread):
    """
    Calculate the Values for the probability Density function across the SPREAD array initiated at top of file

    :param X0: The minimization result array
    :param x_spread: the X values to evaluate the PDF at
    :return: The PDF calculated for each element in the spread array
    """
    X0 = X0.reshape((-1, 3))
    res = np.zeros_like(x_spread)
    # res2 = np.zeros_like(x_spread)

    # dists = [(w, scipy.stats.norm(m, s)) for w, m, s in X0]
    # for i in range(x_spread.shape[0]):
    #     res2[i] = sum(w * d.pdf(x_spread[i]) for w, d in dists)

    for w, mu, sig in X0:
        res += pdf(mu, sig, x_spread) * w
    return res / X0[:, 0].sum(),  # res2


def CDF(X0, x_spread):
    """
    Calculate the Values for the Cumulative density function across the SPREAD array initiated at the top of file

    :param X0: the minimization result array
    :param x_spread: the X values to evaluate the CDF at
    :return: The CDF calculated for each element in the x_spread array
    """
    X0 = X0.reshape((-1, 3))
    dists = [(w, scipy.stats.norm(m, s), scipy.stats.norm(m, s).cdf(0)) for w, m, s in X0]
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


def graphs(m, f_statics, x_spread):
    """
    Plot out the 5 different Charts to visualize the minimization

    :param m: the minimization
    :param f_statics:  the f_statics to compare against
    :param x_spread: the X values to evaluate the curves at
    :return: Nothing
    """
    plt.annotate("Profitable", (0, 10))
    plt.annotate("OverPriced", (0, -10))
    plt.hlines(0, f_statics[:, 0].min(), f_statics[:, 0].max())
    plt.scatter([k for k, pc, _, _ in f_statics if pc],
                [sum(call(k, m, s, 0) * w for w, m, s in m.x) - p
                 for k, pc, p, _ in f_statics if pc],
                c='red')
    plt.scatter([k for k, pc, *_ in f_statics if not pc],
                [sum(put(k, m, s, 0) * w for w, m, s in m.x) - p
                 for k, pc, p, _ in f_statics if not pc],
                c='blue')
    plt.legend(["breakeven", "call", "put"])
    plt.title("Price Differences")
    plt.figure()
    plt.title("Calls")
    plt.scatter([k for k, pc, _, _ in f_statics if pc],
                [sum(call(k, m, s, 0) * w for w, m, s in m.x)
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
    for i in range(1, m.x.shape[0]):
        pfds = PDF(m.x[i - 1:i], x_spread)
        plt.plot(x_spread, -pfds[0])
    plt.plot(x_spread, PDF(m.x, x_spread)[0], lw=5)
    plt.ylim(-.1, .1)

    plt.figure()
    for i in range(1, m.x.shape[0]):
        plt.plot(x_spread, CDF(m.x[:i], x_spread), )
    plt.plot(x_spread, CDF(m.x, x_spread), lw=5)
    l, h = CI(m.x, f_statics)
    print(l, h)
    plt.vlines([l, h], 0, 1, colors='k')
    plt.show(block=True)


def analyze(f_statics, x_spread):
    """
    Load up the minimization and list of intermediate minimizations from file and run analysis on them

    :param f_statics: the f_statics to run against
    :param x_spread the X values to evaluate the fit curves with
    :return: Nothing
    """
    with open('pick.le', 'rb') as fin:
        m = pickle.load(fin)
    # with open('dill.pickle', 'rb') as fin:
    #     ma = pickle.load(fin)
    print(m)
    graphs(m, f_statics, x_spread=x_spread)
    # for mx in ma:
    #     m.x = mx
    #     graphs(m, f_statics)
