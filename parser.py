import pickle
import msgpack
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.stats
from numba import njit, prange


with open('here', 'rb') as f:
    statics = msgpack.unpackb(f.read())

# print(statics[b'GME'][130.0])
spread = np.linspace(50,
                     250,
                     500)


@njit()
def pdf(mu, sig, X):
    return np.exp(-np.power(X-mu, 2)/(2*sig**2))/np.sqrt(2*np.pi*sig**2)


@njit()
def cdf(mu, sig, x, delta):
    DX = np.linspace(x, x+delta)[1]-x
    return pdf(mu, sig, np.linspace(x, x+delta)).sum() * DX


@njit()
def call(strike, mu, sig):
    su = 0.0
    DX = np.linspace(0, max(mu*3, 10), 500)[1]

    for i in np.linspace(0, max(mu*3, 10), 500):
        su += max(0, i-strike)*cdf(mu, sig, i, DX)
    return su


@njit()
def put(strike, mu, sig):
    su = 0.0
    DX = np.linspace(0, max(mu*3, 10), 500)[1]
    for i in np.linspace(0, max(mu*3, 10), 500):
        su += max(0, strike-i)*cdf(mu, sig, i, DX)
    return su


@njit()
def conform(X0):
    return X0.reshape((-1, 3))


@njit(parallel=True)
def bayes_error(X0: np.array, fstatics) -> int:
    X0 = X0.reshape((-1, 3))
    errn = np.zeros(fstatics.shape[0])
    for i in prange(fstatics.shape[0]):
        strike, typ, mark, vol = fstatics[i]
        for j in prange(X0.shape[0]):
            weight, mu, sig = X0[j][0], X0[j][1], X0[j][2]
            if typ:
                errn[i] += call(strike, mu, sig)*weight
            else:
                errn[i] += put(strike, mu, sig)*weight
        errn[i] -= mark
    errn = errn**2
    return np.sqrt(errn.sum())


def error(strike_price: float, typ: int, mark: float, X0: np.array) -> float:
    if typ:
        return (mark-(np.maximum(strike_price-spread, 0)*X0).sum())**2
    else:
        return (mark-(np.maximum(0, spread-strike_price)*X0).sum())**2


def calc_error(X0: np.array, fstatics) -> int:
    err = 0
    result = np.empty(fstatics.shape[0])
    for i in range(fstatics.shape[0]):
        strike_price, typ, mark, vol = fstatics[i]
        result[i] = error(strike_price, typ, mark, X0)
    err = result.sum()
    return err


def static_array(statics, Ticker=b'GME'):
    return np.array([[strike,
                     1.0 if y == b'C' else 0.0,
                     float(data.get(b'mark_price')),
                     float(data.get(b'volume'))]
                    for strike, opt in statics[Ticker].items()
                    for y, data in opt.items()])

# statics


def fn(X, stat):
    # print(stat.shape)

    result = bayes_error(X, stat)

#    return result/20*np.sin(np.pi/4*result/20)

#    return result/np.log(result+1)+np.log(result)
#    if result > 883.925:
    return np.log10(result+1)*100
    return result/3


def main(statics, length=1, X0=None, prev=np.inf):
    if X0 is None:
        X0 = np.empty((0, 3))

    X0 = X0.reshape((-1, 3))
    longer = length-X0.shape[0]
    X1 = np.stack((np.full(longer, 1/length),
                   np.full(longer, 100),
                   np.full(longer, 10)),
                  1)

    X0 = np.vstack((X0, X1))
    con = {'type': 'eq', 'fun': lambda x: 1-x.ravel()[::3].sum()}
    print(bayes_error(X0, statics))
    bou = [(.001, None)
           if i % 3 == 0 else
           (0, None)
           if i % 3 == 1 else
           (.001, None)
           for i in range(length*3)]

    m = scipy.optimize.minimize(fun=bayes_error,
                                args=(statics),
                                x0=X0,
                                constraints=con,
                                bounds=bou,
                                # callback=lambda x: print(
                                #    bayes_error(x, statics)),
                                options={'disp': None,
                                         'iprint': -1,
                                         'eps': 1e-08,
                                         'maxiter': 15000,
                                         'ftol': 2.220446049250313e-09, })

    m.jac = conform(m.jac)
    m.x = conform(m.x)
    print(m)

    if prev/m.fun > 1.1 or prev-m.fun > 20:
        print("="*20)
        mdeeper = main(statics, length+1, m.x, m.fun)
        if mdeeper is not None:
            m = mdeeper
    else:
        return None
    if length == 1:
        with open('pick.le', 'wb') as f:
            pickle.dump(m, f)
    return m


def PDF(X0):
    X0 = X0.reshape((-1, 3))
    res = np.zeros_like(spread)
    res2 = np.zeros_like(spread)

    dists = [(w, scipy.stats.norm(m, s)) for w, m, s in X0]
    for i in range(spread.shape[0]):
        res2[i] = sum(w*d.pdf(spread[i]) for w, d in dists)

    for w, mu, sig in X0:
        res += pdf(mu, sig, spread) * w
    return res/X0[:, 0].sum(), res2


def CDF(X0):
    X0 = X0.reshape((-1, 3))
    dists = [(w, scipy.stats.norm(m, s)) for w, m, s in X0]
    return [sum(w*d.cdf(x) for w, d in dists) for x in spread]


def analyze(statics):
    with open('pick.le', 'rb') as f:
        m = pickle.load(f)
    m.x = m.x.reshape((-1, 3))
    plt.annotate("Profitable", (0,  10))
    plt.annotate("OverPriced", (0, -10))
    plt.hlines(0, statics[:, 0].min(), statics[:, 0].max())
    plt.scatter([k for k, pc, _, _ in statics if pc],
                [sum(call(k, m, s)*w for w, m, s in m.x)-p
                 for k, pc, p, _ in statics if pc],
                c='red')
    plt.scatter([k for k, pc, *_ in statics if not pc],
                [sum(put(k, m, s)*w for w, m, s in m.x)-p
                 for k, pc, p, _ in statics if not pc],
                c='blue')
    plt.legend(["breakeven", "call", "put"])
    plt.title("Price Differences")
    plt.figure()
    plt.title("Calls")
    for k, pc, p, vol in statics:
        if pc:
            plt.scatter(k, p, c="red")
            plt.scatter(k, sum(call(k, m, s)*w for w, m, s in m.x), c='blue')
    print(*m.x.reshape((-1, 3)).tolist(), sep='\n')
    plt.figure()
    for i in range(1, m.x.shape[0]+1):
        pfds = PDF(m.x[i-1:i])
        plt.plot(spread, -pfds[0])
    plt.plot(spread, PDF(m.x)[0], lw=5)
    plt.ylim(-.1, .1)
    plt.figure()
    for i in range(1, m.x.shape[0]+1):
        plt.plot(spread, CDF(m.x[:i]),)
    plt.plot(spread, CDF(m.x), lw=5)
    plt.show(block=True)


if __name__ == "__main__":
    main(static_array(statics, b"GME"))
    analyze(static_array(statics, b'GME'))
