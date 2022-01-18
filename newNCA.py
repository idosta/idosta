from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, ifft, fftshift, ifftshift, fftfreq
import pandas as pd
from matplotlib import pyplot as plt

# dot state description (0, down, up, 1)
# physical parameters

ga = 1.0
ec = 50.0 * ga
nu = 1.0 / ga
beta = 1.0 / ga
V = 40 * ga
miu = array([V / 2, -V / 2])  # 0'th place for left and 1 for right lead
U = 5 * ga
gate = 0
epsilon0 = -U / 2 + gate * ga
E = (0, epsilon0, epsilon0, 2 * epsilon0 + U)
lamb = 0
t_max = 5  # maximal time

# numerical parameters
N = 1001  # number of time points
dt = t_max / (N - 1)
times = linspace(0, t_max, N)
cutoff_factor = 100.0
dw = 0.01
w = arange(-cutoff_factor * ec, cutoff_factor * ec, dw)
d_dyson = 0.00000000001


# define mathematical functions
def fft_integral(x, y):
    return (fftconvolve(x, y)[:len(x)] - 0.5 * (x[:] * y[0] + x[0] * y[:])) * dt


def integral_green(bare_green, old_green, self_energy):
    total = fft_integral(bare_green, self_energy)
    return fft_integral(total, old_green)


def gamma(energy):
    return 0.5 * ga / ((1 + exp(nu * (energy - ec))) * (1 + exp(-nu * (energy + ec))))


def f(energy, mu):
    return 1 / (1 + exp(beta * (energy - mu)))


def de_integral(y, h, k, lim):
 # y - the function, h - precision integral, k - limit of tansinh integral (infinity), lim - limit of original integral
    l = arange(-k, k, h)
    x = tanh(0.5 * pi * sinh(l))
    g = lim * y(x * lim)
    w = 0.5 * h * pi * cosh(l) / (cosh(0.5 * pi * sinh(l))) ** 2
    return sum(w * g)

#
# plt.plot(w, exp(1j * times[100] * w)*gamma(w)*f(w, miu[0]))
# plt.plot(w, exp(1j * times[100] * w)*gamma(w)*(1-f(w, miu[0])))
# plt.show()


delta_l_energy = [-1j * gamma(w) * f(w, miu[0]), -1j * gamma(w) * f(w, miu[1])]
delta_g_energy = [-1j * gamma(w) * (1 - f(w, miu[0])), -1j * gamma(w) * (1 - f(w, miu[1]))]

delta_l_temp = [ifftshift(fft(fftshift(delta_l_energy[0]))) * dw / pi, ifftshift(fft(fftshift(delta_l_energy[1]))) * dw / pi]
delta_g_temp = [ifftshift(fft(fftshift(delta_g_energy[0]))) * dw / pi, ifftshift(fft(fftshift(delta_g_energy[1]))) * dw / pi]


def time_to_fftind(t):
    return int(cutoff_factor * ec / dw + round(t * cutoff_factor * ec / pi))


hl = zeros(N, complex)
hg = zeros(N, complex)
for i in range(N):
    hl[i] = delta_l_temp[0][time_to_fftind(times[i])] + delta_l_temp[1][time_to_fftind(times[i])]
    hg[i] = delta_g_temp[0][time_to_fftind(times[i])] + delta_g_temp[1][time_to_fftind(times[i])]

# r = []
# p = []
# for i in range(N):
# #     r.append(trapz(delta_l_energy[0] * exp(-1j * times[i] * w) / pi, w) + trapz(delta_l_energy[1] * exp(-1j * times[i] * w) / pi, w))
#     p.append(trapz(delta_g_energy[0] * exp(-1j * times[i] * w) / pi, w) + trapz(
#         delta_g_energy[1] * exp(-1j * times[i] * w) / pi, w))


df_guy = pd.read_csv("/home/ido/NCA/Delta_greater.out", " ")
plt.plot(times, df_guy.iloc[1000:, [1]], "+")
plt.plot(times, real(hg), ".")
# plt.plot(times, real(p), "o")
plt.show()
plt.plot(times, df_guy.iloc[1000:, [2]], "+")
plt.plot(times, imag(hg), ".")
plt.plot(times, imag(p), "o")
plt.show()

