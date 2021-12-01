import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import pandas as pd


def QME(tl, v, epsilon, beta, U, ga):
    FI = []
    for a in [0,1]:
        tm = np.sqrt(tl * ga)
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

        def gamma(tb, tm, eps_l, w):
            temp = 0
            if abs(w - eps_l) < (2 * tb):
                temp = ((tm ** 2) / (2 * tb ** 2)) * np.sqrt(4 * tb ** 2 - (w - eps_l) ** 2)
            return temp

        def f(miu, beta, w):
            if w > 0:
                return 1 / (np.exp(beta * (w - miu)) + 1)
            else:
                return 1 - (1 / (np.exp(beta * (-w - miu)) + 1))

        R = [np.zeros((4, 4)), np.zeros((4, 4))]
        for l in range(2):
            for i in range(4):
                for j in range(4):
                    R[l][i, j] = abs(elc[i, j]) * gamma(tl, tm, miu[l], elc[i, j] * (E[i] - E[j])) * f(miu[l], beta,
                                                                                                       (E[i] - E[j]))

        m = np.copy(R)
        for l in range(2):
            for i in range(4):
                m[l][i, i] = -(m[l][0, i] + m[l][1, i] + m[l][2, i] + m[l][3, i])

        p0 = np.zeros(4)
        for i in range(4):
            p0[i] = np.exp(-beta * E[i])
        p0 = p0 / sum(p0)

        def modmatrix(elc, x):
            q = np.zeros((4, 4))
            for i in range(4):
                for j in range(4):
                    q[i, j] = np.exp(x * (elc[i, j]))  # can count total
            return q

        def p(x, t, m, p0, a):
            M = np.zeros((4, 4))
            if a == 1:
                M = (m[0] + m[1] * modmatrix(elc, x))
            if a == 0:
                M = (m[0] * modmatrix(elc, x) + m[1])

            # a=0 for the right lead and a=1 for the left lead
            return np.matmul(expm(M * t), p0)

        def C_cal(t, m, p0, a):
            d = []

            def f(x):
                return np.log(sum(p(x, t, m, p0, a)))

            h = 1
            d.append(0.5 * (f(h) - f(-h)) / h)
            h = h / 10
            d.append(0.5 * (f(h) - f(-h)) / h)
            while d[len(d) - 1] - d[len(d) - 2] > 0.001 * d[len(d) - 1]:
                h = h / 10
                d.append(0.5 * (f(h) - f(-h)) / h)
            return d[len(d) - 1]

        t = []
        t.append(0)
        C = []
        C.append(0)
        I = []
        I.append(0)
        dt = 1
        for i in range(10000):
            t.append(t[len(t) - 1] + dt)
            C.append(C_cal(t[len(t) - 1], m, p0, 0))
            I.append(C[len(C) - 1] / t[len(t) - 1])
        if I[len(I) - 1] > 0:
            while (I[len(C) - 1] - I[len(C) - 2]) > 0.001 * (I[len(C) - 10] - I[len(C) - 11]):
                t.append(t[len(t) - 1] + dt)
                C.append(C_cal(t[len(t) - 1], m, p0, 0))
                I.append(C[len(C) - 1] / t[len(t) - 1])

        FI.append(I[len(C) - 1])
    return FI[0], FI[1]


data = []
beta = 1
U = 40
tlrange = [25]
T = 0
ga = 1
for v in range(-20, 120, 20):
    for epsilon in range(-100, 120, 20):
        for tl in tlrange:
            I = QME(tl, v, epsilon, beta, U, ga)
            data.append([tl, v, epsilon, beta, U, ga, I[0], I[1]])
            print(I)
            T = T + 1
            print(T)

df = pd.DataFrame(data, columns=["tl", "v", "epsilon", "beta", "U", "gamma", "IR", "IL"])
df.to_csv('01122021.csv')
