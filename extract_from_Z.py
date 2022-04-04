from numpy import *
import matplotlib.pyplot as plt
from pathlib import Path

lamb = 0.0001  # remember to update
t_l = 1
t_m = 1
u = 8
dim = 1
T = 0.1


def openfile(h, v, T):
    my_file = Path(
        '/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(
            t_l) + '_t_m'
        + str(t_m) + '_u' + str(u) + '/h' + str(h) + '_v' + str(v) + '/NCA_GF v=' + str(v) + 'h=' + str(h) + '.out')
    if my_file.is_file():
        L = loadtxt(
            '/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(
                t_l) + '_t_m'
            + str(t_m) + '_u' + str(u) + '/h' + str(h) + '_v' + str(v) + '/NCA_GF v=' + str(v) + 'h=' + str(h) + '.out')
        return L, 1
    else:
        print("no file" + str(h) + str(v))
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


def get_c4(z):
    def g(x):  # calculate the partition function log at lambda
        return log(x)

    return g(z[:, 9]) + g(z[:, 1]) - 4 * (g(z[:, 7]) + g(z[:, 3])) + 6 * g(z[:, 5]) / (lamb ** 4)


def get_s(z, m):
    if m == 1:
        c = get_c1(z)
    if m == 2:
        c = get_c2(z)
    if m == 3:
        c = get_c3(z)
    if m == 4:
        c = get_c4(z)
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


V = around(arange(0, 6.2, 0.2), 1)
H = arange(0, 5.25, 0.25)
C = zeros((len(V), len(H)))
N = copy(C)
dC = copy(C)
ddC = copy(C)
dN = copy(C)
NT = copy(C)

i = 0
for v in V:
    j = 0
    for h in H:
        Z = openfile(h, v, T)
        if Z[1] == 1:
            C[i, j] = average(get_current(Z[0])[-3:])
            N[i, j] = average(get_noise(Z[0])[-3:])
            NT[i, j] = average(get_noise_tag(Z[0])[-3:])
            print(h, v)
        j += 1
    i += 1

save('/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' +
     str(t_l) + '_t_m' + str(t_m) + '_u' + str(u) + '/I_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out',
     C)
save('/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' +
     str(t_l) + '_t_m' + str(t_m) + '_u' + str(u) + '/noise_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out',
     N)
save('/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' +
     str(t_l) + '_t_m' + str(t_m) + '_u' + str(u) + '/noise2_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out',
     NT)
