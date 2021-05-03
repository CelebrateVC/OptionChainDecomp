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
import DistributionHandler

ts = ['GOOGL 210604', 'GOOGL 230120', 'GOOGL 210611', 'GOOGL 210716', 'GOOGL 220916', 'GOOGL 210528', 'GOOGL 230616', 'GOOGL 210521', 'GOOGL 210507', 'GOOGL 220121', 'GOOGL 210514', 'GOOGL 210917', 'GOOGL 211015', 'GOOGL 210820', 'GOOGL 210618', 'GOOGL 211217', 'GOOGL 220617']

# Change this to a message-pack file of the correct format (see ReadMe)
with open('Goog Morning.msgpack', 'rb') as f:
    statics = msgpack.unpack(f)
ticker = b'GOOGL 230616'
STOCK_PRICE = 2_377


@njit()
def conform(X0, X1, longer):
    """
    Transform Matrix used in optimizer into what is expected

    :param X0: array with {LONGER} new observations and the remaining weights after that
    :param X1: array of set weights
    :param longer: number of new observations in X0
    :return: array with shape (?,3) in the form (weight,mean,std)
    """
    X_Mu_Sig = np.vstack((X1.reshape((-1, 3))[:, 1:],
                          X0[:2 * longer].reshape((-1, 2))))
    return np.hstack((X0[2 * longer:].reshape((-1, 1)), X_Mu_Sig))


def fn(X, stat, Xp, longer):
    """
    Function called by the optimization

    :param X: The dynamic parts of the optimization array
    :param stat: the f_statics we are building our model on
    :param Xp: the static part of the optimization array
    :param longer: the length of the new items in the array
    :return: the result of calculating the error (with some normalization function on it if needs arise
    """
    X = conform(X, Xp, longer)
    result = DistributionHandler.bayes_error(X, stat,STOCK_PRICE)
    return result
    # return result/20*np.sin(np.pi/4*result/20)
    # return result/np.log(result+1)+np.log(result)
    # return np.log10(result+1)*100
    # return result/3


def main(f_statics, length=1, X0=None, prev=np.inf, ma=[], first =True):
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
    X1 = np.array([x for y in (np.full(longer, 100),
                               np.full(longer, 1),
                               np.full(length, 1 / length)) for x in y]
                  )
    con = {'type': 'eq', 'fun': lambda x: 1 - x.ravel()[2:].sum()}

    bou = [(0, None),
           (.01, None)]*longer + [(0.01, None) for _ in range(length)]

    m = scipy.optimize.minimize(fun=fn,
                                args=(f_statics, X0, longer),
                                x0=X1,
                                constraints=con,
                                bounds=bou,
                                # callback=lambda x: print(
                                #    bayes_error(x, f_statics)),
                                options={'disp': None,
                                         'iprint': -1,
                                         'eps': 1e-08,
                                         'maxiter': 15000,
                                         'ftol': 2.220446049250313e-09, })
    m.jac = conform(m.jac, np.zeros_like(X0), longer)
    m.x = conform(m.x, X0, longer)
    print(m)
    ma.append(m.x)

    if prev / m.fun > 1.1 or (prev - m.fun) / f_statics[:, 0].mean() > .005:
        print("=" * 20)
        mdeeper = main(f_statics, length + 1, m.x, m.fun, ma, False)
        # print(ma)
        if mdeeper is not None:
            m = mdeeper
    else:
        return None
    if first:
        with open('pick.le', 'wb') as fin:
            pickle.dump(m, fin)
        with open('dill.pickle', 'wb') as fin:
            pickle.dump(ma, fin)
    return m


if __name__ == "__main__":
    sta = DistributionHandler.static_array(statics, ticker)
    main(sta, 3)
    DistributionHandler.analyze(sta,STOCK_PRICE)
