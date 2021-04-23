"""
This optimizer will calculate the composite distributions for the option chains selected

The nature of this optimizer will throw out the previously found Mean and Standard Distributions
     optimizing all means, standard divs, and weights at the same time.

the optimizer will exit when the error of the composite does not drop by $2 or by 10%, to prevent overfitting
"""
import pickle
import msgpack
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats
from numba import njit, prange

# Change this to a message-pack file of the correct format (see ReadMe)
with open('here', 'rb') as f:
    statics = msgpack.unpackb(f.read())

# Used for the display after optimizing - NOT Used in optimization
spread = np.linspace(50,
                     250,
                     500)


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
def call(strike, mu, sig):
    """
    Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div

    :param strike: strike price of the option
    :param mu: mean of distribution
    :param sig: standard deviation of distribution
    :return: estimate of call price
    """
    su = 0.0
    DX = np.linspace(0, max(mu * 3, 10), 500)[1]

    for i in np.linspace(0, max(mu * 3, 10), 500):
        su += max(0, i - strike) * cdf(mu, sig, i, DX)
    return su


@njit()
def put(strike, mu, sig):
    """
    Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div

    :param strike: strike price of option
    :param mu: mean of distribution
    :param sig: standard deviation of distribution
    :return: estimate of put price
    """
    su = 0.0
    DX = np.linspace(0, max(mu * 3, 10), 500)[1]
    for i in np.linspace(0, max(mu * 3, 10), 500):
        su += max(0, strike - i) * cdf(mu, sig, i, DX)
    return su


@njit()
def conform(X0):
    """
    Transform Matrix used in optimizer into what is expected

    :param X0: array with {LONGER} new observations and the remaining weights after that
    :return: array with shape (?,3) in the form (weight,mean,std)
    """
    return X0.reshape((-1, 3))


@njit(parallel=True)
def bayes_error(X0: np.array, f_statics) -> int:
    """
    Calculate the Root Sum Squared Weighted Error weighted by the vol column in f_statics

    :param X0: (?, 3) array of form (weight, mean, std)
    :param f_statics: (?, 4) array of form (strike_price, is_call, mark_price, error_weighting_factor)
    :return: Root Sum Squared Weighted Error
    """
    X0 = X0.reshape((-1, 3))
    errn = np.zeros(f_statics.shape[0])
    for i in prange(f_statics.shape[0]):
        strike, typ, mark, vol = f_statics[i]
        for j in prange(X0.shape[0]):
            weight, mu, sig = X0[j][0], X0[j][1], X0[j][2]
            if typ:
                errn[i] += call(strike, mu, sig) * weight
            else:
                errn[i] += put(strike, mu, sig) * weight
        errn[i] -= mark
    errn = errn ** 2
    return np.sqrt(errn.sum())


def static_array(f_statics, Ticker=b'GME'):
    """
    Transform the f_statics dictionary into a numba conformal array

    :param f_statics: the f_statics dictionary loaded from the message pack file
    :param Ticker: the specific ticker to extract from the f_statics dictionary
    :return: np.array containing only floats so it may be passed into Numba decorated functions
    """
    return np.array([[strike,
                      1.0 if y == b'C' else 0.0,
                      float(data.get(b'mark_price')),
                      float(data.get(b'volume'))]
                     for strike, opt in f_statics[Ticker].items()
                     for y, data in opt.items()])


def fn(X, stat):
    """
    Function called by the optimization

    :param X: The dynamic parts of the optimization array
    :param stat: the f_statics we are building our model on
    :return: the result of calculating the error (with some normalization function on it if needs arise
    """
    X = conform(X)
    result = bayes_error(X, stat)
    return result
    # return result/20*np.sin(np.pi/4*result/20)
    # return result/np.log(result+1)+np.log(result)
    # return np.log10(result+1)*100
    # return result/3


def main(f_statics, length=1, X0=None, prev=np.inf, ma=[]):
    """
    Optimize the following f_statics recursively until the error does not change by enough

    :param f_statics: the f_statics to minimize onto
    :param length: the number of items to add to the previous
    :param X0: the previously calculated minimization - to be held static except for the weighting
    :param prev: the previous error result
    :param ma: the full list of minimization arrays to iterate across later
    :return: the minimization result
    """
    if X0 is None:
        X0 = np.empty((0, 3))

    X0 = X0.reshape((-1, 3))
    longer = length - X0.shape[0]
    X1 = np.stack((np.full(longer, 1 / length),
                   np.full(longer, 100),
                   np.full(longer, 10)),
                  1)

    X0 = np.vstack((X0, X1))
    con = {'type': 'eq', 'fun': lambda x: 1 - x.ravel()[::3].sum()}
    print(bayes_error(X0, statics))
    bou = [(.001, None)
           if i % 3 == 0 else
           (0, None)
           if i % 3 == 1 else
           (.001, None)
           for i in range(length * 3)]

    m = scipy.optimize.minimize(fun=bayes_error,
                                args=(f_statics,),
                                x0=X0,
                                constraints=con,
                                bounds=bou,
                                # callback=lambda x: print(
                                #    bayes_error(x, f_statics)),
                                options={'disp': None,
                                         'iprint': -1,
                                         'eps': 1e-08,
                                         'maxiter': 15000,
                                         'ftol': 2.220446049250313e-09, })

    m.jac = conform(m.jac)
    m.x = conform(m.x)
    print(m)
    ma.append(m.x)

    if prev / m.fun > 1.1 or prev - m.fun > 2:
        print("=" * 20)
        m_deeper = main(f_statics, length + 1, m.x, m.fun, ma)
        print(ma)
        if m_deeper is not None:
            m = m_deeper
    else:
        return None
    if length == 1:
        with open('pick.le', 'wb') as fin:
            pickle.dump(m, fin)
        with open('dill.pickle', 'wb') as fin:
            pickle.dump(ma, fin)
    return m


def PDF(X0):
    """
    Calculate the Values for the probability Density function across the SPREAD array initiated at top of file

    :param X0: The minimization result array
    :return: The PDF calculated for each element in the x_spread array
    """
    X0 = X0.reshape((-1, 3))
    res = np.zeros_like(spread)
    # res2 = np.zeros_like(x_spread)

    # dists = [(w, scipy.stats.norm(m, s)) for w, m, s in X0]
    # for i in range(x_spread.shape[0]):
    #     res2[i] = sum(w * d.pdf(x_spread[i]) for w, d in dists)

    for w, mu, sig in X0:
        res += pdf(mu, sig, spread) * w
    return res / X0[:, 0].sum(),  # res2


def CDF(X0, i_spread):
    """
    Calculate the Values for the Cumulative density function across the SPREAD array initiated at the top of file

    :param i_spread:
    :param X0: the minimization result array
    :return: The CDF calculated for each element in the x_spread array
    """
    X0 = X0.reshape((-1, 3))
    dists = [(w, scipy.stats.norm(m, s)) for w, m, s in X0]
    return [sum(w * d.cdf(x) for w, d in dists) for x in i_spread]


def CI(X0):
    """
    Calculate the 5% and 95% confidence interval

    :param X0: the minimization result array
    :return: the 2 confidence intervals
    """

    x_spread = np.linspace(0, 25, 1001)
    five = x_spread[np.array(CDF(X0, x_spread)) < .05][-1]

    x_spread = np.linspace(250, 1000, 7501)
    ninefive = x_spread[np.array(CDF(X0, x_spread)) > .95][0]

    return five, ninefive


def graphs(m, f_statics):
    """
    Plot out the 5 different Charts to visualize the minimization

    :param m: the minimization
    :param f_statics:  the f_statics to compare against
    :return: Nothing
    """
    plt.annotate("Profitable", (0, 10))
    plt.annotate("OverPriced", (0, -10))
    plt.hlines(0, f_statics[:, 0].min(), f_statics[:, 0].max())
    plt.scatter([k for k, pc, _, _ in f_statics if pc],
                [sum(call(k, m, s) * w for w, m, s in m.x) - p
                 for k, pc, p, _ in f_statics if pc],
                c='red')
    plt.scatter([k for k, pc, *_ in f_statics if not pc],
                [sum(put(k, m, s) * w for w, m, s in m.x) - p
                 for k, pc, p, _ in f_statics if not pc],
                c='blue')
    plt.legend(["breakeven", "call", "put"])
    plt.title("Price Differences")
    plt.figure()
    plt.title("Calls")
    plt.scatter([k for k, pc, _, _ in f_statics if pc],
                [sum(call(k, m, s) * w for w, m, s in m.x)
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
        pfds = PDF(m.x[i - 1:i])
        plt.plot(spread, -pfds[0])
    plt.plot(spread, PDF(m.x)[0], lw=5)
    plt.ylim(-.1, .1)

    plt.figure()
    for i in range(1, m.x.shape[0] + 1):
        plt.plot(spread, CDF(m.x[:i], spread), )
    plt.plot(spread, CDF(m.x, spread), lw=5)
    plt.show(block=True)


def analyze(f_statics):
    """
    Load up the minimization and list of intermediate minimizations from file and run analysis on them

    :param f_statics: the f_statics to run against
    :return: Nothing
    """
    with open('pick.le', 'rb') as fin:
        m = pickle.load(fin)
    with open('dill.pickle', 'rb') as fin:
        ma = pickle.load(fin)
    graphs(m, f_statics)
    for mx in ma:
        m.x = mx
        graphs(m, f_statics)


if __name__ == "__main__":
    main(static_array(statics, b"GME"))
    analyze(static_array(statics, b'GME'))
