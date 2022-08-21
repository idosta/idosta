from numpy import *
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as LA
from numpy.linalg import inv
from functions import *


def QME(tl, tm, v, epsilon, T, U, t_max, N, l):
    FI = []

    # physical parameters

    ga = tm ** 2 / tl
    miu = (v * ga / 2, -v * ga / 2)
    U = U * ga
    bet = 1 / (ga * T)
    eps_m = -U / 2 + epsilon * ga
    # E is the energy of the dot=(00,up0,down0,updown)
    E = (0, eps_m, eps_m, 2 * eps_m + U)

    n = (0, 1, 1, 2)
    elc = zeros((4, 4))
    for i in range(4):
        for j in range(4):
            elc[i, j] = (n[i] - n[j])
            if abs(elc[i, j]) > 1.1:
                elc[i, j] = 0
    # numerical parameters
    tim = linspace(0, t_max, N)
    dt = tim[1] - tim[0]

    # initial condition
    p0 = zeros(4)
    for i in range(4):
        p0[i] = exp(-bet * E[i])
    p0 = p0 / sum(p0)
    print(E)

    def gamm(tb, tmm, eps_l, ww):
        if abs(ww - eps_l) < (2 * tb):
            return ((tmm ** 2) / (2 * (tb ** 2))) * sqrt(4 * tb ** 2 - (ww - eps_l) ** 2)
        else:
            return 0

    gam_mat = zeros((4, 4))
    for i in range(4):
        for j in range(4):
            gam_mat[i, j] = gamm(tl, tm, 0, E[j] - E[i])



    def getR(beta, mu, E):

        def gu(n, m):
            return gam_mat[n, m] * f(E[n] - E[m], mu, beta)

        def gd(n, m):
            return gam_mat[n, m] * (1.0 - f(E[m] - E[n], mu, beta))

        R = array(
            [array([0., gu(1, 0), gu(2, 0), 0.]),
             array([gd(0, 1), 0., 0., gu(3, 1)]),
             array([gd(0, 2), 0., 0., gu(3, 2)]),
             array([0., gd(1, 3), gd(2, 3), 0.]),
             ])
        return R

    RL, RR = getR(bet, miu[0], E), getR(bet, miu[1], E)

    R = RL + RR

    def get_M(lam):
        M = zeros((4, 4))
        for i in range(4):
            for j in range(4):
                if i == j:
                    M[i, j] = -sum(R[i, :])
                else:
                    M[i, j] = RR[j, i] + RL[j, i] * exp(lam * elc[i, j])
        return M

    # Generate M matrix:
    def p(t, lam):
        print(get_M(lam))
        return matmul(expm(get_M(lam) * t), p0)

    times = linspace(0, t_max, N)

    P = zeros((4, len(times)))
    for i in range(len(times)):
        P[:, i] = p(times[i], l)

    def get_current(p, m):
        return p[0] * (m[0, 1] + m[0, 2]) + \
               p[1] * (- m[1, 0] + m[1, 3]) + \
               p[2] * (- m[2, 0] + m[2, 3]) + \
               p[3] * (- m[3, 2] - m[3, 1])

    def find_first_c(lamb):
        w, v = LA.eig(get_M(0, lamb))
        m0 = inv(v) @ get_M(0, lamb) @ v
        ml = inv(v) @ get_M(l, lamb) @ v
        dm = (ml - m0) / l
        return dm

    return P


p = zeros((4, 11, 3), complex)
for v in range(-5, 6):
    for h in range(-1, 2):
        p[:, 5 + v, 1 + h] = QME(1, 1, v, h, 1, 1, 10, 1000, 0)[:, -1]

        print(v, h)
save('p_qme', p)

plt.plot()
