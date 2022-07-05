from numpy import *
import matplotlib.pyplot as plt
from scipy.linalg import expm
from numpy import linalg as LA
from numpy.linalg import inv

def QME(tl, tm, v, epsilon, T, U, t_max, N):
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
    l = 0.00001

    # initial condition
    p0 = zeros(4)
    for i in range(4):
        p0[i] = exp(-bet * E[i])
    p0 = p0 / sum(p0)

    def gamm(tb, tmm, eps_l, ww):
        if abs(ww - eps_l) < (2 * tb):
            return ((tmm ** 2) / (2 * tb ** 2)) * sqrt(4 * tb ** 2 - (ww - eps_l) ** 2)
        else:
            return 0

    def fermi(b, mu, en):
        return 1.0 / (exp(b * (en - mu)) + 1)

    def getR(b, mu, En):

        def gu(nn, m):
            return gamm(tl, tm, mu, En[m] - En[nn]) * fermi(b, mu, En[nn] - En[m])

        def gd(nn, m):
            return gamm(tl, tm, mu, En[nn] - En[m]) * (1.0 - fermi(b, mu, En[m] - En[nn]))

        Rt = array(
            [array([0., gu(1, 0), gu(2, 0), 0.]),
             array([gd(0, 1), 0., 0., gu(3, 1)]),
             array([gd(0, 2), 0., 0., gu(3, 2)]),
             array([0., gd(1, 3), gd(2, 3), 0.]),
             ])
        return Rt

    def get_M(lamb, d):
        M = zeros((4, 4))
        for ii in range(4):
            for jj in range(4):
                if ii == jj:
                    M[ii, jj] = sum(R[ii, :])
                else:
                    if d == 0:
                        M[ii, jj] = -RR[jj, ii] - RL[jj, ii] * exp(lamb * elc[ii, jj])
                    if d == 1:
                        M[ii, jj] = -RR[jj, ii] * exp(-lamb * elc[ii, jj]) - RL[jj, ii]
        return M

    def norm(a):
        return a / sum(a)

    def p(times, lamb, d, p00):
        we = zeros((N, 4))
        for ta in range(N):
            we[ta] = expm(-get_M(lamb, d) * times[ta]) @ p00
            we[ta] = we[ta]
        return we

    for a in [0, 1]:
        RL, RR = getR(bet, miu[0], E), getR(bet, miu[1], E)
        R = RL + RR
        Z = zeros((2, N))
        P = p(tim, l, a, p0)
        for i in range(4):
            Z[1] += P[:, i]
        P = p(tim, 0, a, p0)
        for i in range(4):
            Z[0] += P[:, i]
        C = (log(Z[1]) - log(Z[0])) / l
        plt.plot(tim, C)
    plt.show()
    w, v = LA.eig(get_M(0, a))
    m0 = inv(v) @ get_M(0, a) @ v
    ml = inv(v) @ get_M(l, a) @ v
    dm = (ml - m0) / l
    print(dm)
    print((C[N - 1] - C[N - 2]) / (tim[N - 1] - tim[N - 2]))





    # def get_current(p, m):
    #     return p[0] * (m[0, 1] + m[0, 2]) + \
    #            p[1] * (- m[1, 0] + m[1, 3]) + \
    #            p[2] * (- m[2, 0] + m[2, 3]) + \
    #            p[3] * (- m[3, 2] - m[3, 1])

    return FI[0], FI[1]


QME(1, 1, 0.1, 1, 1, 1, 100, 200)

# data = []
# beta = 1
# U = 40
# tlrange = [25]
# T = 0
# tm = 5
# for v in range(-100, 101, 1):
#     for epsilon in range(-100, 105, 5):
#         for tl in tlrange:
#             I = QME(tl, tm, v, epsilon, beta, U)
#             data.append([tl, tm, v, epsilon, beta, U, I[0], I[1], I[2], I[3]])
#             print(I)
#             T = T + 1
#             print(T)
#
# df = pd.DataFrame(data, columns=["tl" , "tm", "v", "epsilon", "beta", "U", "IL", "IR", "IL_DIRECT", "IR_DIRECT"])
# df.to_csv('08122021hr.csv')
