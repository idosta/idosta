from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftshift, ifftshift
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
from numpy.polynomial.chebyshev import chebval
import matplotlib.pyplot as plt

K = load("K.npy")
P0 = load("P.npy")
N = 350
x = 7
Z = zeros((x, N // x), complex)
for i in range(x):
    for t in range(N // x):
        for j in range(4):
            Z[i, t] += P0[:] @ K[j, :, i * N // x + 0, i * N // x + t]
    plt.plot(linspace(0, N // x, len(Z[i, :])), log(Z[i, :]), label=i)
    # plt.plot(linspace(0, N // x, len(Z[i, :])), log(Z[i, :]).imag, label=i)
plt.legend()
plt.show()

