from math import sin, cos


def derivative(x, w):
    # draw the computational graph
    p = x[0] * w[0]
    q = x[1] * w[1]
    l = sin(p)
    m = l * l
    n = cos(q)
    s = 2 + m + n
    f = 1 / s

    # calculate the derivation by computational graph
    df = -1 / (s * s)
    ds = 1
    dm = 2 * l
    dn = -sin(q)
    dl = cos(p)
    dp_x = w[0]
    dp_w = x[0]
    dq_x = w[1]
    dq_w = x[1]

    df_x1 = df * ds * dm * dl * dp_x
    df_w1 = df * ds * dm * dl * dp_w
    df_x2 = df * ds * dn * dq_x
    df_w2 = df * ds * dn * dq_w

    # calculate derivation by hand
    dx1 = -2 * w[0] * sin(x[0] * w[0]) * cos(
        x[0] * w[0]) / (2 + sin(w[0] * x[0])**2 + cos(w[1] * x[1]))**2
    dw1 = -2 * x[0] * sin(x[0] * w[0]) * cos(
        x[0] * w[0]) / (2 + sin(w[0] * x[0])**2 + cos(w[1] * x[1]))**2
    dx2 = w[1] * sin(
        w[1] * x[1]) / (2 + sin(w[0] * x[0])**2 + cos(w[1] * x[1]))**2
    dw2 = x[1] * sin(
        w[1] * x[1]) / (2 + sin(w[0] * x[0])**2 + cos(w[1] * x[1]))**2

    # print the result
    print('derivation of x1 by hand:', dx1)
    print('derivation of x1 by program:', df_x1)
    print('derivation of x2 by hand:', dx2)
    print('derivation of x2 by program:', df_x2)
    print('derivation of w1 by hand:', dw1)
    print('derivation of w1 by program:', df_w1)
    print('derivation of w2 by hand:', dw2)
    print('derivation of w2 by program:', df_w2)


derivative([1, 1], [2, 2])
