import matplotlib.pyplot as plt
from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftshift, ifftshift

v, eps, u, temperature, t_m, t_l = 1, 0, 5, 1, 1, 1

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
# N is number of time points index N is time zero

t_max = 20  # numerical parameter for infinity
N = 10000 * t_max  # number of time data points
times = linspace(-t_max, t_max, 2 * N + 1)
times_plus = linspace(0, t_max, N + 1)
dt = times[1] - times[0]
cutoff_factor = 100.0
N_w = 5000000
w = linspace(- 2 * t_l * cutoff_factor, 2 * t_l * cutoff_factor, N_w)
dw = w[1] - w[0]
d_dyson = 1e-7
a = 0


def fft_integral(x, y):
    temp = (fftconvolve(x, y)[0:len(x)] - 0.5 * (x[0:len(x)] * y[0] + x[0] * y[0:len(x)])) * dt
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
for i in range(N, 2 * N + 1):
    hl[0][i] = conj(delta_l_temp[0][time_to_fftind(times[i])])
    hl[1][i] = conj(delta_l_temp[1][time_to_fftind(times[i])])
    hg[0][i] = delta_g_temp[0][time_to_fftind(times[i])]
    hg[1][i] = delta_g_temp[1][time_to_fftind(times[i])]
    hl[:2, 2 * N - i] = conj(hl[:2, i])
    hg[:2, 2 * N - i] = conj(hg[:2, i])
hl[2] = hl[0] + hl[1]  # lesser hybridization function
hg[2] = hg[0] + hg[1]  # greater


#
# plt.plot(times, hl[2])
# plt.plot(times, hg[2])
# plt.show()
# plt.plot(times, hl[2].imag)
# plt.plot(times, hg[2].imag)
# plt.show()

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
        temp[2, t_se - N] = temp[1, t_se - N]
        temp[3, t_se - N] = hg[2][t_se] * (green[1, t_se - N] + green[2, t_se - N])
    return temp


# build the initial states

G0 = zeros((4, N + 1), complex)
for t in range(N + 1):
    for state_i in range(4):
        G0[state_i, t] = g(times_plus[t], state_i) * exp(-5 * times_plus[t] / t_max)
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
# savetxt("g.out", c_[G.real])
# savetxt("hg.out", c_[hg.real])
# savetxt("hl.out", c_[hl.real])
# savetxt("gi.out", c_[G.imag])
# savetxt("hgi.out", c_[hg.imag])
# savetxt("hli.out", c_[hl.imag])

sG = zeros((4, 2 * N + 1), complex)
sG[:, N:] = G
sG[:, :N + 1] = conj(G[:, ::-1])

# calculating vertex function
G = sG
K = zeros((4, 2 * N + 1), complex)


def normal_k(vertex):
    vertex[:, :] = vertex[:, :] / sum(vertex[:, N])
    return vertex


def mid_term(vertex):
    # this term is the multiplication of the hybridization function and the vertex function
    temp_mat = zeros((4, 2 * N + 1), complex)
    temp_mat[0] = (vertex[1] + vertex[2]) * hl[2]
    temp_mat[3] = (vertex[1] + vertex[2]) * hg[2]
    temp_mat[1] = vertex[0] * hg[2] + vertex[3] * hl[2]
    temp_mat[2] = temp_mat[1]
    return temp_mat


K = normal_k(G)


def update_vertex(M):
    A = zeros((4, 2 * N + 1), complex)
    B = zeros((4, 2 * N + 1), complex)
    MR = M[:, :]
    for c in range(4):
        A[c] = fft_integral(G[c, :], MR[c, :])
        AR = A[:, ::-1].copy()
        B[c] = fft_integral(conj(G[c, :]), AR[c, :])
    # plt.plot(times, A[1])
    # plt.show()
    # plt.plot(times, B[1])
    # plt.show()
    return B


delta_K = d_dyson + 1
C = 0
It = zeros((4, 2 * N + 1), complex)
while delta_K > d_dyson:
    K_old = copy(K)
    It = mid_term(K_old)
    if C % 1 == 0:
        plt.plot(times, K[0], label='0')
        plt.plot(times, K[1], label='1')
        plt.plot(times, K[2], label='2')
        plt.plot(times, K[3], label='3')
        plt.title('K')
        plt.show()
        plt.plot(times, K[2].imag, label='2i')
        plt.plot(times, K[3].imag, label='3i')
        plt.legend()
        plt.title('K')
        plt.show()
        # plt.title('Mid')
        # plt.plot(times, It[0], label='0')
        # plt.plot(times, It[1], label='1')
        # plt.plot(times, It[2], label='2')
        # plt.plot(times, It[3], label='3')
        # plt.show()
        # plt.title('Mid')
        # plt.plot(times, It[2].imag, label='2i')
        # plt.plot(times, It[3].imag, label='3i')
        # plt.legend()
        # plt.show()
    print('K iteration number', C, 'with delta K', delta_K)
    newK = update_vertex(It)
    # sK = newK[:, N:]
    # newK[:, :N + 1] = conj(sK[:, ::-1]).copy()
    K = (1 - a) * normal_k(newK) + a * K
    delta_K = amax(abs(K - K_old))
    C += 1

print("NCA vertex function Converged within", delta_K, "after", C, "iterations.")

plt.plot(times, K[0], label='0')
plt.plot(times, K[1], label='1')
plt.plot(times, K[2].imag, label='1i')
plt.plot(times, K[3].imag, label='0i')
plt.legend()
plt.show()
print(K[0, N], K[1, N], K[2, N], K[3, N])

# def current_cal(vertex, propagator, hyb_g):
#     down = vertex[1] * propagator[0]
#     empt = vertex[0] * propagator[1]
#     I = fft_integral(down[N:], hyb_g[N:]) + fft_integral(empt[N:], hyb_g[N:])
#     # I += fft_integral(down[:N + 1][::-1], hyb_g[:N + 1][::-1]) + fft_integral(empt[:N + 1][::-1], hyb_g[:N + 1][::-1])
#     return I
#
#
# plt.plot(times_plus, current_cal(K, G, hg[0]))
# plt.show()
