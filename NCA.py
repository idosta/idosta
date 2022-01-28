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
t_max = 5  # maximal time

# numerical parameters
N = 201  # number of time points
dt = t_max / (N - 1)
times = linspace(0, t_max, N)
cutoff_factor = 100.0
dw = 0.01
w = linspace(-cutoff_factor * ec, cutoff_factor * ec, int(2 * cutoff_factor * ec / dw) + 1)
d_dyson = 1e-10


# define mathematical functions
def fft_integral(x, y):
    temp = (fftconvolve(x, y)[:N] - 0.5 * (x[:N] * y[0] + x[0] * y[:N])) * dt
    return temp


def integral_green(bare_green, self_energy, old_green):
    total23, total21 = fft_integral(self_energy, old_green), fft_integral(self_energy, bare_green)
    return 0.5 * (fft_integral(bare_green, total23) + fft_integral(old_green, total21))


def gamma(energy):
    return 0.5 * ga / ((1 + exp(nu * (energy - ec))) * (1 + exp(-nu * (energy + ec))))


def f(energy, mu):
    return 1 / (1 + exp(beta * (energy - mu)))


delta_l_energy = [gamma(w) * f(w, miu[0]), gamma(w) * f(w, miu[1])]
delta_g_energy = [gamma(-w) * (1 - f(w, miu[0])), gamma(-w) * (1 - f(w, miu[1]))]

delta_l_temp = [ifftshift(fft(fftshift(delta_l_energy[0]))) * dw / pi,
                ifftshift(fft(fftshift(delta_l_energy[1]))) * dw / pi]
delta_g_temp = [ifftshift(fft(fftshift(delta_g_energy[0]))) * dw / pi,
                ifftshift(fft(fftshift(delta_g_energy[1]))) * dw / pi]


def time_to_fftind(ti):
    return int(cutoff_factor * ec / dw + round(ti * cutoff_factor * ec / pi))


hl = zeros(N, complex)
hg = zeros(N, complex)
for i in range(N):
    hl[i] = conj(delta_l_temp[0][time_to_fftind(times[i])] + delta_l_temp[1][time_to_fftind(times[i])])
    hg[i] = delta_g_temp[0][time_to_fftind(times[i])] + delta_g_temp[1][time_to_fftind(times[i])]


def g(time, site):  # t for time and j for the site number in the dot
    return exp(-1j * E[site] * time)


def update_green(self_energy, old_green, bare_green):
    temp = copy(bare_green)
    for site in range(4):
        temp[site, :] -= integral_green(bare_green[site, :], self_energy[site, :], old_green[site, :])
    return temp


def update_self_energy(number_of_times, green):
    temp = zeros((4, number_of_times), complex)
    for t_se in range(number_of_times):
        temp[0, t_se] = hl[t_se] * (green[1, t_se] + green[2, t_se])
        temp[1, t_se] = hg[t_se] * green[0, t_se] + hl[t_se] * green[3, t_se]
        temp[2, t_se] = hg[t_se] * green[0, t_se] + hl[t_se] * green[3, t_se]
        temp[3, t_se] = hg[t_se] * (green[1, t_se] + green[2, t_se])
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
    delta_G = amax(abs(G - G_old))
    C += 1
    print(C, delta_G)
G = update_green(SE, G_old, G0)
for s in range(4):
    savetxt("/home/ido/NCA/temp_results/G_ido" + str(s) + ".out",
            c_[times, G[s, :].real, G[s, :].imag])
print("NCA green function Converged within", d_dyson, "after", C, "iterations.")


#  calculate the vertex function K

def sign_time(f, t1, t2):
    if t2 - t1 > 0:
        return f[t2 - t1]
    else:
        return conj(f[t1 - t2])


for i in range(N):
    hl[i] = transpose(hl[i]) * exp(-1j * lamb * times[i])
    hg[i] = hg[i] * exp(1j * lamb * times[i])  # check for problems with lambda
H_mat = zeros((4, 4, N, N), complex)
for t1 in range(N):
    for t2 in range(N):
        H_mat[0, 1, t1, t2] = sign_time(hg, t1, t2)
        H_mat[0, 2, t1, t2] = sign_time(hg, t1, t2)
        H_mat[1, 0, t1, t2] = sign_time(hl, t1, t2)
        H_mat[1, 3, t1, t2] = sign_time(hg, t1, t2)
        H_mat[2, 0, t1, t2] = sign_time(hl, t1, t2)
        H_mat[2, 3, t1, t2] = sign_time(hg, t1, t2)
        H_mat[3, 1, t1, t2] = sign_time(hl, t1, t2)
        H_mat[3, 2, t1, t2] = sign_time(hl, t1, t2)

def gen_bare_vertex(final_state, initial_state, final_time, initial_time, green_function):
    return conj(green_function[final_state, final_time]) * green_function[initial_state, initial_time] \
           * int(bool(initial_state == final_state))


def mult_vertex(k):
    v = zeros((4, N, N), complex)
    for l in range(4):
        for m in range(4):
            v[l] += k[m] * H_mat[m, l]
    return v

def update_vertex(p, g, k0):
    temp = copy(k0)
    for f in range(4):
        c = zeros((N, N), complex)
        for t2 in range(N):
            c[:, t2] = fft_integral(p[f, :, t2], g[f, :])
        for t1 in range(N):
            temp[f, t1, :] += fft_integral(conj(g[f, :]), c[t1, :])
    return temp


K0 = zeros((4, 4, N, N), complex)
for j1 in range(N):
    for j2 in range(N):
        for down in range(4):
            K0[down, down, j1, j2] = gen_bare_vertex(down, down, j1, j2, G)
K = copy(K0)
print("start iterations to find the vertex function")

for a in range(4):
    K_old = zeros((4, N, N), complex)
    C = 0
    delta_K = d_dyson + 1
    while delta_K > d_dyson:
        K_old = copy(K[a])
        P = mult_vertex(K[a])
        K[a] = update_vertex(P, G, K0[a])
        delta_K = amax(abs(K[a] - K_old))
        C += 1
        print(a, C, delta_K)



print("NCA vertex function Converged within", d_dyson, "after", C, "iterations.")

P = zeros((4, 4, N), complex)
for i in range(4):
    for j in range(4):
        for tn in range(N):
            P[i, j, tn] = K[i, j, tn, 0]
        savetxt("/home/ido/NCA/temp_results/P_ido" + str(i) + str(j) + ".out", c_[times, P[i, j, :]])

# calculate partition function
Z = []
for jt in range(N):
    temp_Z = 0
    for i in range(4):
        for j in range(4):
            temp_Z += G0[i] * K[j, i, jt, jt]
    Z.append(temp_Z)
