from math import exp
import numpy as np
from numpy import matmul


def sig(x):
    res = []
    for i in range(0, 3):
        temp = 1 / (1 + exp(-x[i]))
        res.append(temp)
    res = np.array(res)
    res = res.reshape(3, 1)
    return res


def div_sig(x):
    res = []
    for i in range(0, 3):
        temp1 = 1 / (1 + exp(-x[i]))
        temp2 = (1 - temp1) * temp1
        # print(tp)
        res.append(temp2)
    res = np.array(res)
    res = res.reshape(3, 1)
    return res


def mat_deriv(x, w):
    # draw the computational graph
    p = matmul(w, x)
    q = sig(p)
    r = np.linalg.norm(q)**2

    # calculate the derivation
    dr = 2 * q
    print('df:', dr)
    dq = div_sig(p)
    print('dq:', dq)
    dp_x = w.T
    dp_w = x.reshape(3, 1)
    print('dr*dq:', dr * dq)

    dr_x = matmul(dp_x, dr * dq)
    dr_w = np.outer(dp_w.T, dr * dq)
    print('dr_x:', dr_x)
    print('dr_w:', dr_w)


x = np.array([1, 0.2, 0.5])
w = np.array([1, 1, 1, 0.5, 0.5, 0.5, 0.2, 0.2, 0.2])
w = w.reshape(3, 3)
print('x:', x)
print('w:', w)
mat_deriv(x, w)
