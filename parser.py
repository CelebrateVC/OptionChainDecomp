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
import DistributionHandler

ts = ['GOOGL 210604', 'GOOGL 230120', 'GOOGL 210611', 'GOOGL 210716', 'GOOGL 220916', 'GOOGL 210528', 'GOOGL 230616',
      'GOOGL 210521', 'GOOGL 210507', 'GOOGL 220121', 'GOOGL 210514', 'GOOGL 210917', 'GOOGL 211015', 'GOOGL 210820',
      'GOOGL 210618', 'GOOGL 211217', 'GOOGL 220617']

# Change this to a message-pack file of the correct format (see ReadMe)
with open('Goog Morning.msgpack', 'rb') as f:
    statics = msgpack.unpack(f)
ticker = b'GOOGL 230616'
STOCK_PRICE = 2_377


@njit()
def conform(X0):
    """
    Transform Matrix used in optimizer into what is expected

    :param X0: array with {LONGER} new observations and the remaining weights after that
    :return: array with shape (?,3) in the form (weight,mean,std)
    """
    return X0.reshape((-1, 3))


def fn(X, stat, cur_stock):
    """
    Function called by the optimization

    :param X: The dynamic parts of the optimization array
    :param stat: the f_statics we are building our model on
    :return: the result of calculating the error (with some normalization function on it if needs arise
    """
    X = conform(X)
    result = DistributionHandler.bayes_error(X, stat, cur_stock)
    return result
    # return result/20*np.sin(np.pi/4*result/20)
    # return result/np.log(result+1)+np.log(result)
    # return np.log10(result+1)*100
    # return result/3


def main(f_statics, length=1, X0=None, prev=np.inf, ma=[], first=True):
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
    print(fn(X0, f_statics, STOCK_PRICE))
    bou = [(.001, None)
           if i % 3 == 0 else
           (0, None)
           if i % 3 == 1 else
           (.05, 1000)
           for i in range(length * 3)]

    m = scipy.optimize.minimize(fun=fn,
                                args=(f_statics, STOCK_PRICE),
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
        m_deeper = main(f_statics, length + 1, m.x, m.fun, ma, False)
        # print(ma)
        if m_deeper is not None:
            m = m_deeper
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
    # print(sta)
    main(sta, 1)
    print("=" * 20, "ResultSet", "=" * 20, sep='\n')
    DistributionHandler.analyze(sta, STOCK_PRICE)
