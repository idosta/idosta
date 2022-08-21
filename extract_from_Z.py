from numpy import *
import matplotlib.pyplot as plt
from pathlib import Path

lamb = 0.0001  # remember to update
t_l = 1
t_m = 1
u = 1
dim = 1
T = 1
t_max = 10
t = linspace(0, t_max, 20 * 10 + 1)


def openfile(h, v, T, t_max, name, alg):
    print(
        '/home/ido/wolfgang/' + alg + '/' + alg + '_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' +
        str(t_l) + '_t_m' + str(t_m) + '_t_max' + str(t_max) + '_u' + str(u) + '/h' + str(h) + '_v' + str(v) + '/' +
        name + '.npy')
    my_file = Path(
        '/home/ido/wolfgang/' + alg + '/' + alg + '_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' +
        str(t_l) + '_t_m' + str(t_m) + '_t_max' + str(t_max) + '_u' + str(u) + '/h' + str(h) + '_v' + str(v) + '/' +
        name + '.npy')
    if my_file.is_file():
        L = load(
            '/home/ido/wolfgang/' + alg + '/' + alg + '_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(
                dim) + '_t_l' +
            str(t_l) + '_t_m' + str(t_m) + '_t_max' + str(t_max) + '_u' + str(u) + '/h' + str(h) + '_v' + str(v) + '/' +
            name + '.npy')
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


V = around(arange(2, 2.1, 0.1), 1)
H = around(arange(2, 2.1, 0.1), 1)
I = zeros((3, len(V), len(H)))
p1 = copy(I)
colors = ['b', 'm', 'r', 'g']
i = 0
for v in V:
    j = 0
    for h in H:
        R1 = openfile(h, v, T, t_max, 'z_nca_0', 'compare')[0]
        R2 = openfile(h, v, T, t_max, 'z_p_nca_0', 'compare')[0]
        R3 = openfile(h, v, T, t_max, 'z_qme_0', 'compare')[0]
        for s in range(4):
            print((R2[s]))
            plt.plot(t, R1[s], color=colors[s], label='NCA')
            plt.plot(t, R2[s], '--', color=colors[s], label='P_NCA')
            # plt.plot(t, R3[s], ':', color=colors[s], label='QME')
        plt.legend()
        plt.show()

    i += 1

for u in range(3):
    plt.imshow(I[u])
    plt.colorbar()
    plt.show()
