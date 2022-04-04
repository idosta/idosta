import matplotlib.pyplot as plt
from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftshift, ifftshift
from math import isnan

v, eps, u, temperature, t_m, t_l = 5, 0, 1, 1, 1, 1

ga = (t_m ** 2) / t_l
epsilon_lead = 0 * ga  # adding elctron energy on the lead
beta = 1.0 / (ga * temperature)
V = v * ga
miu = array([V / 2, -V / 2])  # 0'th place for left and 1 for right lead
U = u * ga
gate = eps * ga
epsilon0 = -U / 2 + gate
E = (0, epsilon0, epsilon0, 2 * epsilon0 + U)
# t_max is maximal time

# numerical parameters
# N is number of time points

t_max = 25  # numerical parameter for infinity
N = 5000 * t_max  # number of time data points
times = linspace(-t_max, t_max, 2 * N + 1)
times_plus = linspace(0, t_max, N + 1)
dt = times[1] - times[0]
cutoff_factor = 10.0
N_w = 500000
w = linspace(- 2 * t_l * cutoff_factor, 2 * t_l * cutoff_factor, N_w)
dw = w[1] - w[0]
d_dyson = 1e-6
a = 0.2


def fft_integral(x, y):
    temp = (fftconvolve(x, y)[:len(x)] - 0.5 * (x[:len(x)] * y[0] + x[0] * y[:len(x)])) * dt
    temp[len(x):] = 0
    return temp


def integral_green(bare_green, self_energy, old_green):
    total23, total21 = fft_integral(self_energy, old_green), fft_integral(self_energy, bare_green)
    return 0.5 * (fft_integral(bare_green, total23) + fft_integral(old_green, total21))


def gamma(w_sp):
    P = zeros(len(w_sp))
    for en in range(len(w_sp)):
        if abs(w_sp[en] - epsilon_lead) < (2 * t_l):
            P[en] = (t_m ** 2 / (2 * t_l ** 2)) * sqrt(4 * t_l ** 2 - (w_sp[en] - epsilon_lead) ** 2)
    return P


def f(energy, mu):
    return 1 / (1 + exp(beta * (energy - mu)))


delta_l_energy = [gamma(w) * f(w, miu[0]), gamma(w) * f(w, miu[1])]
delta_g_energy = [gamma(-w) * (1 - f(w, miu[0])), gamma(-w) * (1 - f(w, miu[1]))]

delta_l_temp = [ifftshift(fft(fftshift(delta_l_energy[0]))) * dw / pi,
                ifftshift(fft(fftshift(delta_l_energy[1]))) * dw / pi]
delta_g_temp = [ifftshift(fft(fftshift(delta_g_energy[0]))) * dw / pi,
                ifftshift(fft(fftshift(delta_g_energy[1]))) * dw / pi]


def time_to_fftind(ti):
    return int(cutoff_factor * 2 * t_l / dw) + (round(ti * cutoff_factor * 2 * t_l / pi))


hl = zeros((3, 2 * N + 1), complex)
hg = zeros((3, 2 * N + 1), complex)
for i in range(2 * N + 1):
    hl[0][i] = conj(delta_l_temp[0][time_to_fftind(times[i])])
    hl[1][i] = conj(delta_l_temp[1][time_to_fftind(times[i])])
    hg[0][i] = delta_g_temp[0][time_to_fftind(times[i])]
    hg[1][i] = delta_g_temp[1][time_to_fftind(times[i])]
hl[2] = hl[0] + hl[1]
hg[2] = hg[0] + hg[1]


def g(time, site):  # t for time and j for the site number in the dot
    return exp(-1j * E[site] * time)


def update_green(self_energy, old_green, bare_green):
    temp = copy(bare_green)
    for site in range(4):
        temp[site, :] -= integral_green(bare_green[site, :], self_energy[site, :], old_green[site, :])
    return temp


def update_self_energy(green):
    temp = zeros((4, N + 1), complex)
    for t_se in range(N, 2 * N + 1):
        temp[0, t_se - N] = hl[2][t_se] * (green[1, t_se - N] + green[2, t_se - N])
        temp[1, t_se - N] = hg[2][t_se] * green[0, t_se - N] + hl[2][t_se] * green[3, t_se - N]
        temp[2, t_se - N] = hg[2][t_se] * green[0, t_se - N] + hl[2][t_se] * green[3, t_se - N]
        temp[3, t_se - N] = hg[2][t_se] * (green[1, t_se - N] + green[2, t_se - N])
    return temp


# build the initial states

G0 = zeros((4, N + 1), complex)
for t in range(N + 1):
    for state_i in range(4):
        G0[state_i, t] = g(times_plus[t], state_i) * exp(-10 * times_plus[t] / t_max)
G = copy(G0)
SE = update_self_energy(G)
delta_G = d_dyson + 1
# print("start iterations to calculate G (green function), SE (self energy)")
C = 0
while delta_G > d_dyson:
    G_old = copy(G)
    G = update_green(SE, G_old, G0)
    SE = update_self_energy(G)
    delta_G = amax(abs(G - G_old))
    C += 1
    print('G iteration number', C, 'with delta G', delta_G)
print("NCA green function Converged within", delta_G, "after", C, "iterations.")

sG = zeros((4, 2 * N + 1), complex)
sG[:, N:] = G
sG[:, :N + 1] = conj(G[:, ::-1])
plt.plot(times, sG[0].imag, label='0i')
plt.plot(times, sG[1].imag, label='1i')
plt.plot(times, sG[2], label='1')
plt.plot(times, sG[3], label='0')
plt.legend()
plt.show()
#   calculating vertex function
G = sG
K = zeros((4, 2 * N + 1), complex)


def normal_k(vertex):
    vertex[:, :] = vertex[:, :] / (vertex[0, N] + vertex[1, N] + vertex[2, N] + vertex[3, N])
    return vertex


def mid_term(vertex):
    temp_mat = zeros((4, 2 * N + 1), complex)
    temp_mat[0] = (vertex[1] + vertex[2]) * hl[2]
    temp_mat[3] = (vertex[1] + vertex[2]) * hg[2]
    temp_mat[1] = vertex[0] * hg[2] + vertex[3] * hl[2]
    temp_mat[2] = vertex[0] * hg[2] + vertex[3] * hl[2]
    return temp_mat[::-1]


for i in range(N + 1):
    K[:, N + i] = G[:, N - 1] * conj(G[:, N - 1 - i])
    K[:, N - i] = conj(G[:, N - 1]) * G[:, N - 1 - i]

K = normal_k(K)


def update_vertex(M):
    A = zeros((4, 2 * N + 1), complex)
    B = copy(A)
    for c in range(4):
        A[c] = fft_integral(G[c, :], M[c, :])
        B[c] = fft_integral(conj(G[c, :]), A[c, ::-1])
    return B


delta_K = d_dyson + 1
C = 0
It = zeros((4, 2 * N + 1), complex)
while delta_K > d_dyson:
    K_old = copy(K)
    It = mid_term(K_old)
    print('K iteration number', C, 'with delta K', delta_K)
    K = (1 - a) * update_vertex(It) + a * K
    K = normal_k(K)

    delta_K = amax(abs(K - K_old))
    C += 1

print("NCA vertex function Converged within", delta_K, "after", C, "iterations.")

plt.plot(times, K[0], label='0')
plt.plot(times, K[1], label='1')
plt.plot(times, K[2].imag, label='1i')
plt.plot(times, K[3].imag, label='0i')
plt.legend()
plt.show()
print(K[0, N], K[1, N])

def current_cal(vertex, propagator, hyb_g):
    down = vertex[1] * propagator[1]
    empt = vertex[0] * propagator[0]
    I = fft_integral(down[N:], hyb_g[N:]) + fft_integral(empt[N:], hyb_g[N:])
    # I += fft_integral(down[:N + 1][::-1], hyb_g[:N + 1][::-1]) + fft_integral(empt[:N + 1][::-1], hyb_g[:N + 1][::-1])
    return I


plt.plot(times_plus, current_cal(K, G, hg[2]))
plt.show()






