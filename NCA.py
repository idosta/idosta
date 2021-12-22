import pandas as pd
from numpy import *
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve

# dot state description (0, down, up, 1)
# physical parameters

ga = 1.0
ec = 50.0 * ga
nu = 1.0 / ga
beta = 1.0 / ga
V = 40.0 * ga
miu = array([-V / 2, V / 2])  # 0'th place for left and 1 for right lead
U = 5.0 * ga
gate = 0
epsilon0 = -U / 2 + gate * ga
E = (0, epsilon0, epsilon0, 2 * epsilon0 + U)
lamb = 0
t_max = 5.0  # maximal time

# numerical parameters
Nx = 10000  # number of points for hybridization integral
nec = 100  # limit for the hyb integral function
d_dyson = 0.0000000001
N = 1001  # number of time points
dt = t_max / (N-1)
times = linspace(0, t_max, N)


# define mathematical functions
def integral_green(bare_green, old_green, self_energy):
    total = fftconvolve(bare_green, self_energy)[:N] * dt
    return fftconvolve(total, old_green)[:N] * dt


def gamma(energy):
    return 0.5 * ga / ((1 + exp(nu * (energy - ec))) * (1 + exp(-nu * (energy + ec))))


def f(energy, mu):
    return 1 / (1 + exp(beta * (energy - mu)))


def hyb_lesser(mu):
    temp = zeros(N, complex)
    for t_hl in range(N):
        x = linspace(-nec, nec, Nx)
        y = exp(-1j * x * times[t_hl]) * gamma(x) * f(x, mu) / pi
        temp[t_hl] = trapz(y, x)
    return temp


hl = array([hyb_lesser(miu[0]), hyb_lesser(miu[1])])


def hyb_greater(mu):
    temp = zeros(N, complex)
    for t_hg in range(N):
        x = linspace(-nec, nec, Nx)
        y = exp(1j * x * times[t_hg]) * gamma(-x) * (1 - f(x, mu)) / pi
        temp[t_hg] = trapz(y, x)
    return temp


hg = array([hyb_greater(miu[0]), hyb_greater(miu[1])])


def d_op(spin, role, final_state, initial_state):  # 0=spin down 1=spin up, 0=annihilation 1=creation
    temp = 0
    if spin == 0:
        if role == 0:
            if final_state == 0 and initial_state == 1:
                temp = 1
            if final_state == 1 and initial_state == 3:
                temp = 1
        if role == 1:
            if initial_state == 0 and final_state == 1:
                temp = 1
            if initial_state == 1 and final_state == 3:
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
    temp = 0
    for spin in [0, 1]:
        for lead in [0, 1]:
            temp = temp + (
                    hl[lead, t_cbh] * d_op(spin, 0, down_index, up_index) * d_op(spin, 1, up_index, down_index)
                    * exp(-1j * lamb * times[t_cbh]) + hg[lead, t_cbh] * d_op(spin, 1, down_index, up_index) *
                    d_op(spin, 0, up_index, down_index)) * exp(1j * lamb * times[t_cbh])
    return temp


CBH = zeros((4, 4, N, N), complex)
for down in range(4):
    for up in range(4):
        for it in range(N):
            for ft in range(N):
                CBH[down, up, it, ft] = cross_branch_hyb(down, up, it - ft)


def g(time, site):  # t for time and j for the site number in the dot
    return exp(-1j * E[site] * time)


def update_green(self_energy, old_green, bare_green):
    temp = copy(bare_green)
    for site in range(4):
        temp[site, :] -= integral_green(bare_green[site, :], old_green[site, :], self_energy[site, :])
    return temp


def update_self_energy(number_of_times, green):
    temp = zeros((4, number_of_times), complex)
    for site in range(4):
        for t_se in range(number_of_times):
            for spin in [0, 1]:
                for lead in [0, 1]:
                    for beta_site in range(4):
                        temp[site, t_se] += (hl[lead, t_se] * d_op(spin, 0, site, beta_site) *
                                             d_op(spin, 1, beta_site, site) + hg[lead, t_se] *
                                             d_op(spin, 1, site, beta_site) *
                                             d_op(spin, 0, beta_site, site)) * green[beta_site, t_se]
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
# start iterations to calculate G (green function), SE (self energy)
C = 0
while delta_G > d_dyson:
    G_old = copy(G)
    G = update_green(SE, G_old, G0)
    SE = update_self_energy(N, G)
    delta_G = amax(G - G_old)
    C += 1
    print(".")
df1 = pd.DataFrame(G.T)
df1.columns = ["G0", "G1", "G2", "G3"]
df1.insert(0, "time", times)
df1.to_csv('/home/ido/NCA/temp_results/G.csv')
print("NCA green function Converged within", d_dyson, "after", C, "iterations.")
plt.plot(times, G[0, :].real)
plt.plot(times, G[1, :].real)
plt.plot(times, G[2, :].real)
plt.plot(times, G[3, :].real)
plt.show()
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
        conv_in[:, t_inner] = fftconvolve(multi[b, :, t_inner], conj(green[b, :]))[:N] * dt
    for t_outer in range(N):
        integral[:, t_outer] = fftconvolve(green[b, :], conv_in[:, t_outer])[:N] * dt
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
print("NCA vertex function Converged within", d_dyson, "after", C, "iterations.")

P = zeros((4, 4, N), complex)
for i in range(4):
    for j in range(4):
        for tn in range(N):
            P[i, j, tn] = K[i, j, tn, tn]
        df2 = pd.DataFrame(P[i, j, :].T)
        df2.columns = ["P{}{}".format(i, j)]
        df2.insert(0, "time", times)
        df2.to_csv('/home/ido/NCA/temp_results/P{}_{}.csv'.format(i, j))

# calculate partition function
Z = []
for jt in range(N):
    temp_Z = 0
    for i in range(4):
        for j in range(4):
            temp_Z += G0[i] * K[j, i, jt, jt]
    Z.append(temp_Z)
