from typing import Tuple, Union, Iterable

from numpy import *
from numpy import ndarray
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
from numpy.polynomial.chebyshev import chebval
from matplotlib import pyplot as plt

# physical parameters
t = 1
tl = t  # hoping energy
tm = sqrt(t)
dim = 2

# numerical parameters
ga = 1.0
ec = 2 * t
nu = 1.0 / ga
cutoff_factor = 10
dw = 0.01 * t
w = arange(-ec * cutoff_factor, ec * cutoff_factor, dw)
n_w = len(w)


def gen_hyb(dim, t_mol, t_lead, epsilon, NU, n_c, n_w):
    #     n_c = 70  number of chebyshev coefficients
    #     n_w = 10000   number of energy points
    # NU = number of sites
    N = int(NU ** (1 / dim))

    def build_h_5d(n, ep, tb):
        hamiltonian = sp.lil_matrix((n ** 4, n ** 4))
        for i in range(n ** 4):
            hamiltonian[i, i] = ep
            if i % N != 0:  # nearest in same row
                hamiltonian[i - 1, i] = tb
                hamiltonian[i, i - 1] = tb
            if i - N + 1 > 0:  # previous row
                hamiltonian[i, i - N] = tb
                hamiltonian[i - N, i] = tb
            if i - N ** 2 + 1 > 0:  # previous plane
                hamiltonian[i, i - N ** 2] = tb
                hamiltonian[i - N ** 2, i] = tb
            if i - N ** 3 + 1 > 0:  # previous cube
                hamiltonian[i, i - N ** 3] = tb
                hamiltonian[i - N ** 3, i] = tb
            if i - N ** 4 + 1 > 0:  # previous 4d cube
                hamiltonian[i, i - N ** 4] = tb
                hamiltonian[i - N ** 4, i] = tb
        return hamiltonian.tocsr()

    def build_h_4d(n, ep, tb):
        hamiltonian = sp.lil_matrix((n ** 4, n ** 4))
        for i in range(n ** 4):
            hamiltonian[i, i] = ep
            if i % N != 0:  # nearest in same row
                hamiltonian[i - 1, i] = tb
                hamiltonian[i, i - 1] = tb
            if i - N + 1 > 0:  # previous row
                hamiltonian[i, i - N] = tb
                hamiltonian[i - N, i] = tb
            if i - N ** 2 + 1 > 0:  # previous plane
                hamiltonian[i, i - N ** 2] = tb
                hamiltonian[i - N ** 2, i] = tb
            if i - N ** 3 + 1 > 0:  # previous cube
                hamiltonian[i, i - N ** 3] = tb
                hamiltonian[i - N ** 3, i] = tb
        return hamiltonian.tocsr()

    def build_h_3d(n, ep, tb):
        hamiltonian = sp.lil_matrix((n ** 3, n ** 3))
        for i in range(n ** 3):
            hamiltonian[i, i] = ep
            if i % N != 0:  # nearest in same row
                hamiltonian[i - 1, i] = tb
                hamiltonian[i, i - 1] = tb
            if i - N + 1 > 0:  # previous row
                hamiltonian[i, i - N] = tb
                hamiltonian[i - N, i] = tb
            if i - N ** 2 + 1 > 0:  # previous plane
                hamiltonian[i, i - N ** 2] = tb
                hamiltonian[i - N ** 2, i] = tb
        return hamiltonian.tocsr()

    def build_h_2d(n, ep, tb):
        hamiltonian = sp.lil_matrix((n ** 2, n ** 2))
        for i in range(n ** 2):
            hamiltonian[i, i] = ep
            if i % N != 0:  # nearest in same row
                hamiltonian[i - 1, i] = tb
                hamiltonian[i, i - 1] = tb
            if i - N + 1 > 0:  # previous row
                hamiltonian[i, i - N] = tb
                hamiltonian[i - N, i] = tb
        return hamiltonian.tocsr()

    def build_h_1d(n, ep, tb):
        hamiltonian = sp.lil_matrix((n, n))
        for i in range(n):
            hamiltonian[i, i] = ep
        for i in range(n - 1):
            hamiltonian[i, i + 1] = tb
            hamiltonian[i + 1, i] = tb

        return hamiltonian.tocsr()

    def build_h(n, ep, t_l):
        if dim == 1:
            return build_h_1d(n, ep, t_l), dim
        if dim == 2:
            return build_h_2d(n, ep, t_l), dim
        if dim == 3:
            return build_h_3d(n, ep, t_l), dim
        if dim == 4:
            return build_h_4d(n, ep, t_l), dim
        if dim == 5:
            return build_h_5d(n, ep, t_l), dim

    H, d = build_h(N, epsilon, t_lead)
    E_max = float(eigsh(H, 1, which='LA', return_eigenvectors=False))
    E_min = float(eigsh(H, 1, which='SA', return_eigenvectors=False))
    a = (E_max - E_min) / 2
    b = (E_max + E_min) / 2
    H, d = build_h(N, (epsilon - b) / a, t / a)

    c = zeros((3, N ** d))
    c[0, 0] = 1
    cz = copy(c[0, :])
    c[1] = H @ c[0]
    mu = zeros(n_c)
    mu[0] = 0.5
    mu[1] = c[0, :] @ c[1, :]
    for i in range(2, n_c):
        c[2, :] = 2 * H @ c[1, :] - c[0, :]
        mu[i] = cz @ c[2, :]
        c[0, :] = copy(c[1, :])
        c[1, :] = copy(c[2, :])
    w_sp = linspace(-1, 1, n_w)
    D = chebval(w_sp, mu) * (2 / pi) / sqrt(1 - w_sp ** 2)
    w_sp = w_sp * a + b
    D = D / a
    D = D * pi * t_mol ** 2
    return w_sp, D


epsilon = 0


def gamma(w_sp):
    g = zeros(len(w_sp))
    for i in range(len(w_sp)):
        if abs(w_sp[i] - epsilon) < (2 * t):
            g[i] = (tm ** 2 / (2 * t ** 2)) * sqrt(4 * t ** 2 - (w_sp[i] - epsilon) ** 2)
    return g


def gamma_e(energy):
    return 0.5 * ga / ((1 + exp(nu * (energy - ec))) * (1 + exp(-nu * (energy + ec))))


def gamma_c(energy):
    gam = gen_hyb(dim, tm, tl, 0, 10000, 300, 1000)
    y = zeros(len(energy))
    for i in range(len(energy)):
        if min(gam[0]) < energy[i] < max(gam[0]):
            ind = argmin(abs(gam[0] - energy[i]))
            y[i] = gam[1][ind]
        else:
            y[i] = 0
    return y


# F = gen_hyb(2, 1, 1, -1, 1000, 100, 1000)
plt.plot(w, gamma(w), label="1D")
# plt.plot(F[0], F[1], label="1D")
plt.plot(w, gamma_c(w), label="cal")

plt.legend()
plt.show()

# def f(energy, mu):
#     return 1 / (1 + exp((energy - mu)))
#
#
# delta_l_energy = [gamma_c(w) * f(w, -5), gamma_c(w) * f(w, -1)]
# plt.plot(w, delta_l_energy[0])
# plt.show()
