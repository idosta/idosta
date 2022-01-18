from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, ifft, fftshift, ifftshift
import pandas as pd

# dot state description (0, down, up, 1)
# physical parameters

ga = 1.0
ec = 50.0 * ga
nu = 1.0 / ga
beta = 1.0 / ga
V = 40.0 * ga
miu = array([V / 2, -V / 2])  # 0'th place for left and 1 for right lead
U = 5 * ga
gate = 0
epsilon0 = -U / 2 + gate * ga
E = (0, epsilon0, epsilon0, 2 * epsilon0 + U)
lamb = 0
t_max = 5.0  # maximal time

# numerical parameters
N = 1001  # number of time points
dt = t_max / (N - 1)
times = linspace(0, t_max, N)
cutoff_factor = 5.0
dw = 0.001
w = linspace(-cutoff_factor * ec, cutoff_factor * ec, int(2 * cutoff_factor * ec / dw) + 1)
d_dyson = 0.00000000001


# define mathematical functions
def fft_integral(x, y):
    temp = (fftconvolve(x, y)[:len(x)] - 0.5 * (x[:] * y[0] + x[0] * y[:])) * dt
    temp[(len(x) + 1):] = 0
    return temp


def integral_green(bare_green, self_energy, old_green):
    total23, total21 = fft_integral(self_energy, old_green), fft_integral(self_energy, bare_green)
    return 0.5 * (fft_integral(bare_green, total23) + fft_integral(old_green, total21))


def gamma(energy):
    return 0.5 * ga / ((1 + exp(nu * (energy - ec))) * (1 + exp(-nu * (energy + ec))))


def f(energy, mu):
    return 1 / (1 + exp(beta * (energy - mu)))


delta_l_energy = [-1j * gamma(w) * f(w, miu[0]), -1j * gamma(w) * f(w, miu[1])]
delta_g_energy = [-1j * gamma(w) * (1 - f(w, miu[0])), -1j * gamma(w) * (1 - f(w, miu[1]))]

delta_l_temp = [ifftshift(fft(fftshift(delta_l_energy[0]))) * dw / pi,
                ifftshift(fft(fftshift(delta_l_energy[1]))) * dw / pi]
delta_g_temp = [ifftshift(fft(fftshift(delta_g_energy[0]))) * dw / pi,
                ifftshift(fft(fftshift(delta_g_energy[1]))) * dw / pi]


def time_to_fftind(t):
    return int(cutoff_factor * ec / dw + round(t * cutoff_factor * ec / pi))


hl = zeros(N, complex)
hg = zeros(N, complex)
for i in range(N):
    hl[i] = delta_l_temp[0][time_to_fftind(times[i])] + delta_l_temp[1][time_to_fftind(times[i])]
    hg[i] = delta_g_temp[0][time_to_fftind(times[i])] + delta_g_temp[1][time_to_fftind(times[i])]


def d_op(spin, role, final_state, initial_state):  # 0=spin down 1=spin up, 0=annihilation 1=creation
    temp = 0
    if spin == 0:
        if role == 0:
            if final_state == 0 and initial_state == 1 or final_state == 1 and initial_state == 3:
                temp = 1
        if role == 1:
            if initial_state == 0 and final_state == 1 or initial_state == 1 and final_state == 3:
                temp = 1
    if spin == 1:
        if role == 0:
            if final_state == 0 and initial_state == 2 or final_state == 2 and initial_state == 3:
                temp = 1
        if role == 1:
            if initial_state == 0 and final_state == 2 or initial_state == 2 and final_state == 3:
                temp = 1
    return temp


def cross_branch_hyb(down_index, up_index, t_cbh):
    tempo = 0
    for spin in [0, 1]:
        tempo = tempo - 1j * hl[t_cbh] * d_op(spin, 0, down_index, up_index) * exp(-1j * lamb * times[t_cbh]) + \
                1j * hg[t_cbh] * d_op(spin, 1, down_index, up_index) * exp(1j * lamb * times[t_cbh])
    return tempo


CBH = zeros((4, 4, N, N), complex)
P = zeros((4, 4, N), complex)
for down in range(4):
    for up in range(4):
        for dif in range(-N, N):  # must fix negative times
            P[down, up, dif] = cross_branch_hyb(down, up, dif)
        for it in range(N):
            for ft in range(N):
                CBH[down, up, it, ft] = P[down, up, it - ft]


def g(time, site):  # t for time and j for the site number in the dot
    return exp(-1j * E[site] * time)


def update_green(self_energy, old_green, bare_green):
    temp = copy(bare_green)
    for site in range(4):
        temp[site, :] += integral_green(bare_green[site, :], self_energy[site, :], old_green[site, :])
    return temp


def update_self_energy(number_of_times, green):
    temp = zeros((4, number_of_times), complex)
    for t_se in range(number_of_times):
        temp[0, t_se] = -1j * hl[t_se] * (green[1, t_se] + green[2, t_se])
        temp[1, t_se] = -1j * hg[t_se] * green[0, t_se] - 1j * hl[t_se] * green[3, t_se]
        temp[2, t_se] = -1j * hg[t_se] * green[0, t_se] - 1j * hl[t_se] * green[3, t_se]
        temp[3, t_se] = -1j * hg[t_se] * (green[1, t_se] + green[2, t_se])
    return temp


# build the initial states


G0 = zeros((4, N), complex)
SE = zeros((4, N), complex)
for t in range(N):
    for state_i in range(4):
        G0[state_i, t] = g(times[t], state_i)
G = copy(G0)
SE = update_self_energy(N, G)
delta_G = d_dyson + 1
print("start iterations to calculate G (green function), SE (self energy)")
C = 0
while delta_G > d_dyson:
    G_old = copy(G)
    G = update_green(SE, G_old, G0)
    SE = update_self_energy(N, G)
    delta_G = amax(G - G_old)
    C += 1
    print(".")
for s in range(4):
    savetxt("/home/ido/NCA/temp_results/G_ido" + str(s) + ".out",
            c_[times, G[s, :].real, G[s, :].imag])
print("NCA green function Converged within", d_dyson, "after", C, "iterations.")


#  calculate the vertex function K


def gen_bare_vertex(final_state, initial_state, final_time, initial_time, green_function):
    return conj(green_function[final_state, final_time]) * green_function[initial_state, initial_time] \
           * int(bool(initial_state == final_state))


def integrate_vertex(a, b, green, old_vertex):
    conv_in = zeros((N, N), complex)
    integral = zeros((N, N), complex)
    multi = zeros((4, N, N), complex)
    for at in range(4):
        multi[b] += old_vertex[a, at, :, :] * CBH[at, b, :, :]
    for t_inner in range(N):
        conv_in[:, t_inner] = fft_integral(multi[b, :, t_inner], conj(green[b, :]))
    for t_outer in range(N):
        integral[:, t_outer] = fft_integral(green[b, :], conv_in[:, t_outer])
    return integral


def update_vertex(old_vertex, bare_vertex, green):
    temp = copy(bare_vertex)
    for a in range(4):
        for b in range(4):
            temp[a, b] += integrate_vertex(a, b, green, old_vertex)
    return temp


K0 = zeros((4, 4, N, N), complex)
for j1 in range(N):
    for j2 in range(N):
        for down in range(4):
            K0[down, down, j1, j2] = gen_bare_vertex(down, down, j1, j2, G)

K = copy(K0)
delta_K = d_dyson + 1
print("start iterations to find the vertex function")
C = 0
while delta_K > d_dyson:
    K_old = copy(K)
    K = update_vertex(K_old, K0, G)
    delta_K = amax(K_old - K)
    C += 1
    print(".")
print("NCA vertex function Converged within", d_dyson, "after", C, "iterations.")

P = zeros((4, 4, N), complex)
for i in range(4):
    for j in range(4):
        for tn in range(N):
            P[i, j, tn] = K[i, j, tn, 0]
        savetxt("/home/ido/NCA/temp_results/P_ido" + str(i) + str(j) + ".out", c_[times, P[i, j, :].real])

# calculate partition function
Z = []
for jt in range(N):
    temp_Z = 0
    for i in range(4):
        for j in range(4):
            temp_Z += G0[i] * K[j, i, jt, jt]
    Z.append(temp_Z)
