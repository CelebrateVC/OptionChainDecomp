"""
This optimizer will calculate the composite distributions for the option chains selected

The nature of the optimizer will keep the previously found Mean and Standard Distributions
     while optimizing the new Mean and SD along with all of the weights of the distributions.

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
with open('GME   230120.msgpack', 'rb') as f:
    statics = msgpack.unpackb(f.read())

# Used for the display after optimizing - NOT Used in optimization
spread = np.linspace(0,
                     700,
                     1000)


@njit()
def pdf(mu, sig, X):
    """Probability Density function for the Gaussian Normal Distribution"""
    return np.exp(-np.power(X - mu, 2) / (2 * sig ** 2)) / np.sqrt(2 * np.pi * sig ** 2)


@njit()
def cdf(mu, sig, x, delta):
    """Simple Quad integration of the PDF between X and X+Delta"""
    DX = np.linspace(x, x + delta)[1] - x
    return pdf(mu, sig, np.linspace(x, x + delta)).sum() * DX


@njit()
def call(strike, mu, sig):
    """Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div"""
    su = 0.0
    DX = np.linspace(0, max(mu * 3, 10), 500)[1]

    for i in np.linspace(0, max(mu * 3, 10), 500):
        su += max(0, i - strike) * cdf(mu, sig, i, DX)
    return su


@njit()
def put(strike, mu, sig):
    """Calculate The Discrete Price Estimate for a Call for an underlying with given Mean and ST Div"""
    su = 0.0
    DX = np.linspace(0, max(mu * 3, 10), 500)[1]
    for i in np.linspace(0, max(mu * 3, 10), 500):
        su += max(0, strike - i) * cdf(mu, sig, i, DX)
    return su


@njit()
def conform(X0, X1, longer):
    """Transform Matrix used in optimizer into what is expected"""
    X_Mu_Sig = np.vstack((X1.reshape((-1, 3))[:, 1:],
                          X0[:2 * longer].reshape((-1, 2))))
    return np.hstack((X0[2 * longer:].reshape((-1, 1)), X_Mu_Sig))


@njit(parallel=True)
def bayes_error(X0: np.array, fstatics) -> int:
    """
    Calculate the Root Sum Squared Weighted Error weighted by the vol column in fstatics
    """
    errn = np.zeros(fstatics.shape[0])
    for i in prange(fstatics.shape[0]):
        strike, typ, mark, vol = fstatics[i]
        for j in prange(X0.shape[0]):
            weight, mu, sig = X0[j][0], X0[j][1], X0[j][2]
            if typ:
                errn[i] += call(strike, mu, sig) * weight
            else:
                errn[i] += put(strike, mu, sig) * weight
        errn[i] -= mark
        errn[i] = errn[i] ** 2 * vol
    errn = errn / fstatics[:, 3].sum()
    return np.sqrt(errn.sum())


def static_array(statics, Ticker=b'GME') -> np.array:
    """
    Transform the statics dictionary into a numba conformal array

    :param statics: the statics dictionary loaded from the message pack file
    :param Ticker: the specific ticker to extract from the statics dictionary
    :return: np.array containing only floats so it may be passed into Numba decorated functions
    """
    return np.array([[strike,
                      1.0 if y == b'C' else 0.0,
                      float(data.get(b'mark_price')),
                      np.log(float(data.get(b'open_interest')) + 1)]
                     for strike, opt in statics[Ticker].items()
                     for y, data in opt.items()])


# statics


def fn(X, stat, Xp, longer):
    """
    Function called by the optimization

    :param X: The dynamic parts of the optimiziation array
    :param stat: the statics we are building our model on
    :param Xp: the static part of the optimization array
    :param longer: the length of the new items in the array
    :return: the result of calculating the error (with some normalization function on it if needs arise
    """
    X = conform(X, Xp, longer)
    result = bayes_error(X, stat)
    return result
    # return result/20*np.sin(np.pi/4*result/20)
    # return result/np.log(result+1)+np.log(result)
    # return np.log10(result+1)*100
    # return result/3


def main(statics, length=1, X0=None, prev=np.inf, ma=[]):
    """
    Optimize the following statics recursively until the error does not change by enough

    :param statics: the statics to minimize onto
    :param length: the number of items to add to the previous
    :param X0: the previously calculated minimization - to be held static except for the weighting
    :param prev: the previous error result
    :param ma: the full list of minimization arrays to itterate across later
    :return: the minimization result
    """
    if X0 is None:
        X0 = np.empty((0, 3))

    X0 = X0.reshape((-1, 3))
    longer = length - X0.shape[0]
    X1 = np.array([x for y in (np.full(longer, 100),
                               np.full(longer, 1),
                               np.full(length, 1 / length)) for x in y]
                  )
    con = {'type': 'eq', 'fun': lambda x: 1 - x.ravel()[2:].sum()}

    bou = [(0, None),
           (.01, None)] + [(0.01, None) for _ in range(length)]

    m = scipy.optimize.minimize(fun=fn,
                                args=(statics, X0, longer),
                                x0=X1,
                                constraints=con,
                                bounds=bou,
                                # callback=lambda x: print(
                                #    bayes_error(x, statics)),
                                options={'disp': None,
                                         'iprint': -1,
                                         'eps': 1e-08,
                                         'maxiter': 15000,
                                         'ftol': 2.220446049250313e-09, })
    m.jac = conform(m.jac, np.zeros_like(X0), longer)
    m.x = conform(m.x, X0, longer)
    print(m)
    ma.append(m.x)

    if prev / m.fun > 1.1 or prev - m.fun > 2:
        print("=" * 20)
        mdeeper = main(statics, length + 1, m.x, m.fun, ma)
        print(ma)
        if mdeeper is not None:
            m = mdeeper
    else:
        return None
    if length == 1:
        with open('pick.le', 'wb') as f:
            pickle.dump(m, f)
        with open('dill.pickle', 'wb') as f:
            pickle.dump(ma, f)
    return m


def PDF(X0):
    """
    Calculate the Values for the probability Density function across the SPREAD array initiated at top of file

    :param X0: The minimization result array
    :return: The PDF calculated for each element in the spread array
    """
    X0 = X0.reshape((-1, 3))
    res = np.zeros_like(spread)
    # res2 = np.zeros_like(spread)

    # dists = [(w, scipy.stats.norm(m, s)) for w, m, s in X0]
    # for i in range(spread.shape[0]):
    #     res2[i] = sum(w * d.pdf(spread[i]) for w, d in dists)

    for w, mu, sig in X0:
        res += pdf(mu, sig, spread) * w
    return res / X0[:, 0].sum(),  # res2


def CDF(X0):
    """
    Calculate the Values for the Cumulative density function across the SPREAD array initiated at the top of file

    :param X0: the minimization result array
    :return: The CDF calculated for each element in the spread array
    """
    X0 = X0.reshape((-1, 3))
    dists = [(w, scipy.stats.norm(m, s)) for w, m, s in X0]
    return [sum(w * d.cdf(x) for w, d in dists) for x in spread]


def CI(X0):
    """
    Calculate the 5% and 95% confidence interval

    :param X0: the minimization result array
    :return: the 2 confidence intervals
    """

    spread = np.linspace(0, 25, 1001)
    five = spread[np.array(CDF(X0)) < .05][-1]

    spread = np.linspace(250, 1000, 7501)
    ninefive = spread[np.array(CDF(X0)) > .95][0]

    return five, ninefive


def graphs(m, statics):
    """
    Plot out the 5 different Charts to visualize the minimization

    :param m: the minimization
    :param statics:  the statics to compare against
    :return: Nothing
    """
    plt.annotate("Profitable", (0, 10))
    plt.annotate("OverPriced", (0, -10))
    plt.hlines(0, statics[:, 0].min(), statics[:, 0].max())
    plt.scatter([k for k, pc, _, _ in statics if pc],
                [sum(call(k, m, s) * w for w, m, s in m.x) - p
                 for k, pc, p, _ in statics if pc],
                c='red')
    plt.scatter([k for k, pc, *_ in statics if not pc],
                [sum(put(k, m, s) * w for w, m, s in m.x) - p
                 for k, pc, p, _ in statics if not pc],
                c='blue')
    plt.legend(["breakeven", "call", "put"])
    plt.title("Price Differences")
    plt.figure()
    plt.title("Calls")
    plt.scatter([k for k, pc, _, _ in statics if pc],
                [sum(call(k, m, s) * w for w, m, s in m.x)
                 for k, pc, p, _ in statics if pc],
                c='red')
    plt.scatter([k for k, pc, *_ in statics if pc],
                [p
                 for k, pc, p, _ in statics if pc],
                c='blue')
    plt.legend(['Estimated', 'Actual Price'])
    print(*m.x.reshape((-1, 3)).tolist(), sep='\n')

    plt.figure()
    plt.scatter(statics[:, 0], statics[:, -1])
    plt.title("Volume")

    plt.figure()
    for i in range(1, m.x.shape[0] + 1):
        pfds = PDF(m.x[i - 1:i])
        plt.plot(spread, -pfds[0])
    plt.plot(spread, PDF(m.x)[0], lw=5)
    plt.ylim(-.1, .1)

    plt.figure()
    for i in range(1, m.x.shape[0] + 1):
        plt.plot(spread, CDF(m.x[:i]), )
    plt.plot(spread, CDF(m.x), lw=5)
    plt.show(block=True)


def analyze(statics):
    """
    Load up the minimization and list of intermediate minimizations from file and run analysis on them

    :param statics: the statics to run against
    :return: Nothing
    """
    with open('pick.le', 'rb') as f:
        m = pickle.load(f)
    with open('dill.pickle', 'rb') as f:
        ma = pickle.load(f)
    graphs(m, statics)
    for mx in ma:
        m.x = mx
        graphs(m, statics)


if __name__ == "__main__":
    main(static_array(statics, b"GME"))
    analyze(static_array(statics, b'GME'))
