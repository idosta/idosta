from numpy import *
from matplotlib import pyplot as plt
dt = 1
N = 10
# def t_g_integral(bare_green, old_green, self_energy):
#     total = 0
#     for dt1 in range(0, N):
#         for dt2 in range(0, dt1):
#             total += bare_green[N - 1 - dt1] * self_energy[dt1 - dt2] * old_green[dt2]
#     return total * dt ** 2
x = arange(0, 100)
# y = arange(0, 10000)
# z = arange(0, 10000)
# print(x)
# print(t_g_integral(x, y, z))


def sym(x):
    X = zeros(2 * len(x))
    for i in range(2 * len(x)):
        if i < len(x):
            X[i] = x[i]
        if i >= len(x):
            X[i] = x[2 * len(x) - i - 1]
    return X
plt.plot(x, (fft.ifft(fft.fft(x))))
plt.show