from numpy import *
import matplotlib.pyplot as plt
from pathlib import Path

lamb = 0.000001  # remember to update
t_l = 1
t_m = 1
u = 8
dim = 1
T = 0.1


def openfile(h, v, T):
    my_file = Path(
        '/home/ido/gcohenlab/qme/qme_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(
            t_l) + '_t_m'
        + str(t_m) + '_u' + str(u) + '/h' + str(h) + '_v' + str(v) + '/QME_GF v=' + str(v) + 'h=' + str(h) + '.out')
    if my_file.is_file():
        L = loadtxt(
            '/home/ido/gcohenlab/qme/qme_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(
                t_l) + '_t_m'
            + str(t_m) + '_u' + str(u) + '/h' + str(h) + '_v' + str(v) + '/QME_GF v=' + str(v) + 'h=' + str(h) + '.out')
        return L, 1
    else:
        print("error")
        return 0, 0


def get_c1(z):
    def g(x):  # calculate the partition function log at lambda
        return log(x)

    return ((g(z[:, 9]) - g(z[:, 1])) / (4 * lamb) + (g(z[:, 7]) - g(z[:, 3])) / (2 * lamb)) / 2


def get_c2(z):
    def g(x):  # calculate the partition function log at lambda
        return log(x)

    return ((g(z[:, 7]) + g(z[:, 3]) - 2 * g(z[:, 5])) / (lamb ** 2) +
            (g(z[:, 9]) + g(z[:, 1]) - 2 * g(z[:, 5])) / ((2 * lamb) ** 2)) / 2


def get_c3(z):
    def g(x):  # calculate the partition function log at lambda
        return log(x)

    return (g(z[:, 9]) - g(z[:, 1]) - 2 * g(z[:, 7]) + 2 * g(z[:, 3])) / (2 * lamb ** 3)


def get_current(z):
    c = get_c1(z)
    I = zeros(len(c))
    for i in range(1, len(c) - 1):
        I[i] = (c[i + 1] - c[i - 1]) / (z[i + 1, 0] - z[i - 1, 0])
    I[0] = I[1]
    I[len(c) - 1] = I[len(c) - 2]
    return I


def get_noise(z):
    c = get_c2(z)
    S = zeros(len(c))
    for i in range(1, len(c) - 1):
        S[i] = (c[i + 1] + c[i - 1]) / (z[i + 1, 0] - z[i - 1, 0])
    S[0] = S[1]
    S[len(c) - 1] = S[len(c) - 2]
    return S


def get_noise_tag(z):
    c = get_c3(z)
    S = zeros(len(c))
    for i in range(1, len(c) - 1):
        S[i] = (c[i + 1] + c[i - 1]) / (z[i + 1, 0] - z[i - 1, 0])
    S[0] = S[1]
    S[len(c) - 1] = S[len(c) - 2]
    return S


V = around(arange(-9.09, 10.01, 0.02), 2)
C = zeros((len(V), 1))
N = copy(C)
dC = copy(C)
ddC = copy(C)
dN = copy(C)
NT = copy(C)

i = 0
for v in V:
    j = 0
    for h in arange(0, 0.5, 0.5):
        Z = openfile(h, v, T)
        if Z[1] == 1:
            C[i, j] = average(get_current(Z[0])[-5:])
            N[i, j] = average(get_noise(Z[0])[-5:])
            NT[i, j] = average(get_noise_tag(Z[0])[-5:])
            print(h, v)
        j += 1
    i += 1
plt.plot(V, C.T[0])
plt.show()
plt.matshow(C.T)
plt.show()
plt.matshow(dC.T)
plt.show()
save(
    '/home/ido/gcohenlab/qme/qme_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l) + '_t_m'
    + str(t_m) + '_u' + str(u) + '/I_map_tl=' + str(t_l) + '_tm=' + str(t_m) + '.out',
    C)
save(
    '/home/ido/gcohenlab/qme/qme_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l) + '_t_m'
    + str(t_m) + '_u' + str(u) + '/noise_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out',
    N)
save(
    '/home/ido/gcohenlab/qme/qme_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l) + '_t_m'
    + str(t_m) + '_u' + str(u) + '/noise2_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out',
    NT)
