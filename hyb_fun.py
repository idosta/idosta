from numpy import *
from scipy.sparse.linalg import eigsh
from scipy import sparse
from numpy.polynomial.chebyshev import chebval
from matplotlib import pyplot as plt

# physical parameters
N = 100  # number of sites per lead
epsilon = 10  # energy per electron
t = 1  # hoping energy
# numerical parameters
n_c = 100  # number of chebyshev coefficients
n_w = 1000  # number of energy points


def build_h_1d(n, ep, tb):
    hamiltonian = zeros((n, n))
    for i in range(n):
        hamiltonian[i, i] = ep
    for i in range(n - 1):
        hamiltonian[i, i + 1] = tb
        hamiltonian[i + 1, i] = tb
    return hamiltonian


H = build_h_1d(N, epsilon, t)
E_max = eigsh(H, 1, which='LA', return_eigenvectors=False)
E_min = eigsh(H, 1, which='SA', return_eigenvectors=False)
a = (E_max - E_min) / 1.99999
b = (E_max + E_min) / 2
H = build_h_1d(N, (epsilon - b) / a, t / a)

c = zeros((n_c, N))
c[0, 0] = 1
c[1] = dot(H, c[0])
for i in range(2, n_c):
    c[i] = 2 * dot(H, c[i - 1, :]) - c[i - 2, :]
mu = zeros(n_c)
for i in range(1, n_c):
    mu[i] = dot(c[0, :], c[i, :])
mu[0] = 0.5
w = linspace(-1, 1, n_w)
D = chebval(w, mu) * (2 / pi) / sqrt(1 - w**2)
w = w * a + b
D = D / a


def gamma(w):
    g = zeros(len(w))
    for i in range(len(w)):
        if abs(w[i] - epsilon) < (2 * t):
            g[i] = (1 / (2 * pi * t ** 2)) * sqrt(4 * t ** 2 - (w[i] - epsilon) ** 2)
    return g


plt.plot(w,gamma(w), '--', label="Analytical")
plt.plot(w, D)
plt.show()
