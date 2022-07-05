from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftshift, ifftshift
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
from numpy.polynomial.chebyshev import chebval
import matplotlib.pyplot as plt
from functions import *

K = load("K.npy")
Kg = load("kss.npy")
print(shape(K))
N = 501
x = 1
Z = zeros((x, N // x), complex)
for i in range(0, x):
    for t in range(N // x):
        for j in range(4):
            Z[i, t] += sum(K[j, :, i * N // x + t, i * N // x + t])
    plt.plot(linspace(0, N // x, len(Z[i, :])), log(Z[i, :]), label=i)
    # plt.plot(linspace(0, N // x, len(Z[i, :])), log(Z[i, :]).imag, label=i)
print(log(Z[0, 500]) - log(Z[0, 480]))
Zg = zeros((x, N // x), complex)
# for i in range(0, x):
#     for t in range(N // x):
#         for j in range(4):
#             Zg[i, t] += P0[:] @ Kg[j, :, i * N // x + t, i * N // x + t]
#     plt.plot(linspace(0, N // x, len(Zg[i, :])), log(Zg[i, :]), label="g" + str(i), linewidth=4.0)
plt.legend()
plt.show()
