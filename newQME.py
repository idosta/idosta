import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pandas as pd
from scipy.linalg import eig
from pylab import *


def QME(tl ,tm, v, epsilon, beta, U):
    FI = []

    #phisical parameters

    ga=tm**2/tl
    miu = (v * ga / 2, -v * ga / 2)
    U = U * ga
    beta = beta / ga
    eps_m = -U / 2 + epsilon * ga
    # E is the energy of the dot=(00,up0,down0,updown)
    E = (0, eps_m, eps_m, 2 * eps_m + U)

    n = (0, 1, 1, 2)
    elc = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            elc[i, j] = (n[i] - n[j])
            if abs(elc[i, j]) > 1.1:
                elc[i, j] = 0
    #numerical parameters

    dt = 1
    tmax = 50
    ddt=0.01

    #initial conditioin
    p0 = np.zeros(4)
    for i in range(4):
        p0[i] = np.exp(-beta * E[i])
    p0 = p0 / sum(p0)

    for a in [0, 1]:

        def gamma(tb, tm, eps_l, w):
            if abs(w - eps_l) < (2 * tb):
                return ((tm ** 2) / (2 * tb ** 2)) * np.sqrt(4 * tb ** 2 - (w - eps_l) ** 2)
            else:
                return 0

        def fermi(beta, mu, e):
            return 1.0 / (np.exp(beta * (e - mu)) + 1)

        def getR(beta, mu, E):

            def gu(n, m):
                return gamma(tl, tm, mu, E[m] - E[n]) * fermi(beta, mu, E[n] - E[m])

            def gd(n, m):
                return gamma(tl, tm, mu, E[n] - E[m]) * (1.0 - fermi(beta, mu, E[m] - E[n]))

            R = array( \
                [array([0., gu(1, 0), gu(2, 0), 0.]),
                 array([gd(0, 1), 0., 0., gu(3, 1)]),
                 array([gd(0, 2), 0., 0., gu(3, 2)]),
                 array([0., gd(1, 3), gd(2, 3), 0.]),
                 ])
            return R

        RL, RR = getR(beta, miu[0], E), getR(beta, miu[1], E)
        R = RL + RR

        # Generate M matrix:
        def get_M(lamb,a):
            M = zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    if i == j:
                        M[i, j] = sum(R[i, :])
                    else:
                        if a==0:
                            M[i, j] = -RR[j, i] - RL[j, i] * exp(lamb * elc[i, j])
                        if a==1:
                            M[i, j] = -RR[j, i] * exp(-lamb * elc[i, j]) - RL[j, i]
            return M

#        def p(t, lamb, a, p0):
#            return np.matmul(expm(-get_M(lamb, a) * t), p0)

        def pdt(tm, lamb, a, p0):
            M = get_M(lamb, a)
            P = p0.copy()
            Ps = [p0]
            times = arange(0.0, tm, ddt)

            for t in times[1:]:
                oldP = P.copy()
                for i in range(4):
                    P[i] = oldP[i] - ddt * dot(M[i, :], oldP)
                Ps.append(P.copy())
            return Ps[len(Ps)-1]

        def C_cal(t, p0, a):
            d = []

            def g(lamb):
                return np.log(sum(pdt(t, lamb, a, p0)))
            h = 0.001
            return 0.5 * (g(h) - g(-h)) / h

        t = [tmax]
        C = [0]
        I = [0]

        t.append(t[len(t) - 1] + dt)
        C.append(C_cal(t[len(t) - 1], p0, a))
        I.append(C[len(C) - 1] / t[len(t) - 1])
        t.append(t[len(t) - 1] + 1 + dt)
        C.append(C_cal(t[len(t) - 1], p0, a))
        I.append(C[len(C) - 1] / t[len(t) - 1])


        FI.append(I[len(C) - 1])

    def get_current(p, m):
        return p[0] * (m[0, 1] + m[0, 2]) + \
               p[1] * (- m[1, 0] + m[1, 3]) + \
               p[2] * (- m[2, 0] + m[2, 3]) + \
               p[3] * (- m[3, 2] - m[3, 1])

    return FI[0], FI[1], get_current(pdt(tmax, 0, 0, p0), RL), get_current(pdt(tmax, 0, 1, p0), -RR)


data = []
beta = 1
U = 40
tlrange = [25]
T = 0
tm = 5
for v in range(-100, 101, 1):
    for epsilon in range(-100, 105, 5):
        for tl in tlrange:
            I = QME(tl, tm, v, epsilon, beta, U)
            data.append([tl, tm, v, epsilon, beta, U, I[0], I[1], I[2], I[3]])
            print(I)
            T = T + 1
            print(T)

df = pd.DataFrame(data, columns=["tl" , "tm", "v", "epsilon", "beta", "U", "IL", "IR", "IL_DIRECT", "IR_DIRECT"])
df.to_csv('08122021hr.csv')
