import matplotlib.pyplot as plt
from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftshift, ifftshift

v, eps, u, temperature, lamb, t_max, dim_l, t_m, t_l = 1, 0, 0.2, 1, 0, 15, 1, 1, 1
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

N = 20 * t_max  # number of time data points
times = linspace(0, t_max, N + 1)
dt = times[1] - times[0]
cutoff_factor = 100.0
N_w = 1000000
w = linspace(- 2 * t_l * cutoff_factor, 2 * t_l * cutoff_factor, N_w)
dw = w[1] - w[0]
d_dyson = 1e-6
a = 0.2

#  find the hybridization functions
def f(energy, mu):
    return 1 / (1 + exp(beta * (energy - mu)))


def gamma(w_sp):
    P = zeros(len(w_sp))
    for en in range(len(w_sp)):
        if abs(w_sp[en] - epsilon_lead) < (2 * t_l):
            P[en] = (t_m ** 2 / (2 * t_l ** 2)) * sqrt(4 * t_l ** 2 - (w_sp[en] - epsilon_lead) ** 2)
    return P


def time_to_fftind(ti):
    return int(cutoff_factor * 2 * t_l / dw) + (round(ti * cutoff_factor * 2 * t_l / pi))


gam_w = gamma(w)
delta_l_energy = [gam_w * f(w, miu[0]), gam_w * f(w, miu[1])]

delta_g_energy = [gam_w * (1 - f(w, miu[0])), gam_w * (1 - f(w, miu[1]))]
delta_l_temp = [ifftshift(fft(fftshift(delta_l_energy[0]))) * dw / pi,
                ifftshift(fft(fftshift(delta_l_energy[1]))) * dw / pi]

delta_g_temp = [ifftshift(fft(fftshift(delta_g_energy[0]))) * dw / pi,
                ifftshift(fft(fftshift(delta_g_energy[1]))) * dw / pi]

hl = zeros((3, N + 1), complex)
hg = zeros((3, N + 1), complex)
for i in range(N + 1):
    hl[0][i] = conj(delta_l_temp[0][time_to_fftind(times[i])])
    hl[1][i] = conj(delta_l_temp[1][time_to_fftind(times[i])])
    hg[0][i] = delta_g_temp[0][time_to_fftind(times[i])]
    hg[1][i] = delta_g_temp[1][time_to_fftind(times[i])]
hl[2] = hl[0] + hl[1]
hg[2] = hg[0] + hg[1]


#  calculate the propagators

def fft_integral(x, y):
    temp = (fftconvolve(x, y)[:N + 1] - 0.5 * (x[:N + 1] * y[0] + x[0] * y[:N + 1])) * dt
    temp[N + 1:] = 0
    return temp


def integral_green(bare_green, self_energy, old_green):
    total23, total21 = fft_integral(self_energy, old_green), fft_integral(self_energy, bare_green)
    return 0.5 * (fft_integral(bare_green, total23) + fft_integral(old_green, total21))


def g(ti, site):  # t for time and j for the site number in the dot
    return exp(-1j * E[site] * ti)


def update_green(self_energy, old_green, bare_green):
    temp = copy(bare_green)
    for site in range(4):
        temp[site, :] -= integral_green(bare_green[site, :], self_energy[site, :], old_green[site, :])
    return temp


def update_self_energy(number_of_times, green):
    temp = zeros((4, number_of_times), complex)
    for t_se in range(number_of_times):
        temp[0, t_se] = hl[2][t_se] * (green[1, t_se] + green[2, t_se])
        temp[1, t_se] = hg[2][t_se] * green[0, t_se] + hl[2][t_se] * green[3, t_se]
        temp[2, t_se] = hg[2][t_se] * green[0, t_se] + hl[2][t_se] * green[3, t_se]
        temp[3, t_se] = hg[2][t_se] * (green[1, t_se] + green[2, t_se])
    return temp


G0 = zeros((4, N + 1), complex)
for t in range(N + 1):
    for state_i in range(4):
        G0[state_i, t] = g(times[t], state_i)
G = copy(G0)
SE = update_self_energy(N + 1, G)
delta_G = d_dyson + 1
# print("start iterations to calculate G (green function), SE (self energy)")
C = 0
while delta_G > d_dyson:
    G_old = copy(G)
    G = update_green(SE, G_old, G0)
    SE = update_self_energy(N + 1, G)
    delta_G = amax(abs(G - G_old))
    C += 1
    print(C)


# calculating non_eq vertex function


def sign_time(fun, t_i, t_f):
    if t_f - t_i >= 0:
        return fun[t_f - t_i]
    else:
        return conj(fun[t_i - t_f])


# savetxt("/home/ido/NCA/temp_results/Delta_lesserI.out", c_[times, imag(hl), -real(hl)])
# savetxt("/home/ido/NCA/temp_results/Delta_greaterI.out", c_[times, imag(hg), -real(hg)])
# FCS time
for i in range(N + 1):
    hl[2][i] = hl[0][i] * exp(lamb / 2) + hl[1][i]
    hg[2][i] = hg[0][i] * exp(-lamb / 2) + hg[1][i]
H_mat = zeros((4, 4, N + 1, N + 1), complex)
for t1 in range(N + 1):
    for t2 in range(N + 1):
        H_mat[0, 1, t1, t2] = sign_time(hl[2], t1, t2)
        H_mat[0, 2, t1, t2] = sign_time(hl[2], t1, t2)
        H_mat[1, 0, t1, t2] = sign_time(hg[2], t1, t2)
        H_mat[1, 3, t1, t2] = sign_time(hl[2], t1, t2)
        H_mat[2, 0, t1, t2] = sign_time(hg[2], t1, t2)
        H_mat[2, 3, t1, t2] = sign_time(hl[2], t1, t2)
        H_mat[3, 1, t1, t2] = sign_time(hg[2], t1, t2)
        H_mat[3, 2, t1, t2] = sign_time(hg[2], t1, t2)

K0 = zeros((4, 4, N + 1, N + 1), complex)
for j1 in range(N + 1):
    for j2 in range(N + 1):
        for i in range(4):
            K0[i, i, j1, j2] = conj(G[i, j1]) * G[i, j2]


def mult_vertex(k):
    va = zeros((4, N + 1, N + 1), complex)
    for q in range(4):
        for m in range(4):
            va[q] += k[m] * H_mat[m, q]
    return va


def update_vertex(p, gr, k0):
    temp = copy(k0)
    for i_f in range(4):
        c = zeros((N + 1, N + 1), complex)
        for t_2 in range(N + 1):
            c[:, t_2] = fft_integral(p[i_f, :, t_2], conj(gr[i_f, :]))
        for t_1 in range(N + 1):
            temp[i_f, t_1, :] += fft_integral((gr[i_f, :]), c[t_1, :])
    return temp


K = copy(K0)

for a in range(4):
    C = 0
    delta_K = d_dyson + 1
    while delta_K > d_dyson:
        K_old = copy(K[a])
        P = mult_vertex(K[a])
        K[a] = update_vertex(P, G, K0[a])
        delta_K = amax(abs(K[a] - K_old))
        C += 1
        print(C, delta_K)

p0 = zeros(4)
for i in range(4):
    p0[i] = exp(-E[i] * beta)
p0 = p0 / sum(p0)

Pr = zeros((4, N + 1), complex)
for i in range(4):
    for tn in range(N + 1):
        Pr[i, tn] = K[i, :, tn, tn] @ p0[:]
plt.plot(times, Pr[0])
plt.plot(times, Pr[0].imag)
plt.plot(times, Pr[1])
plt.plot(times, Pr[1].imag)
plt.show()

c = log(sum(K[:, 0, 9 * N // 10, N]) / sum(K[:, 0, 8 * N // 10, 9 * N // 10])) / (
        times[9 * N // 10] - times[8 * N // 10])
print(c)

#  steady state NCA
M = N


def build_negative_times(y):
    full_time = linspace(-t_max, t_max, 2 * M - 1)
    z = zeros(2 * M - 1, complex)
    z[M - 1:] = y
    z[:M] = conj(y[::-1])
    return full_time, z


def conj_maker(A):
    hA = A[:M // 2 + 1]
    F = zeros(M + 1, complex)
    F[N // 2:] = copy(hA)
    F[:N // 2 + 1] = copy(conj(hA[::-1]))
    return F


times = linspace(-t_max / 2, t_max / 2, N + 1)
hL = conj_maker(hl[2])
hG = conj_maker(hg[2])


def cf(time, co):
    return exp(co * time)


def mid_term(vertex):
    # this term is the multiplication of the hybridization function and the vertex function
    temp_mat = zeros((4, 4, N + 1), complex)
    temp_mat[:, 0] = (vertex[:, 1] + vertex[:, 2]) * hL * cf(times, -c)
    temp_mat[:, 3] = (vertex[:, 1] + vertex[:, 2]) * hG * cf(times, -c)
    temp_mat[:, 1] = vertex[:, 0] * hG + vertex[:, 3] * hL * cf(times, -c)
    temp_mat[:, 2] = temp_mat[:, 1] * cf(times, -c)
    return temp_mat


def update_vertex(V):
    A = zeros((4, 4, 4, N + 1), complex)
    B = zeros((4, 4, N + 1), complex)
    for bet in range(4):
        for alpha in range(4):
            for de in range(4):
                A[bet, alpha, de] = fft_integral(G[bet, :], V[de, alpha, :])
                AR = copy(A[:, :, :, ::-1]) * cf(times, c)
                B[bet, alpha] += fft_integral(conj(G[de, :]), AR[bet, alpha, de, :])
    return B


K0 = zeros((4, 4, N + 1), complex)
for i in range(4):
    for j1 in range(N + 1):
        K0[i, i, j1] = conj(G[i, j1]) * G[i, N]
    K0[i, i, :] = conj_maker(K0[i, i, :])
K = copy(K0)
delta_K = d_dyson + 1
C = 0
while delta_K > d_dyson:
    K_old = copy(K)
    print('K iteration number', C, 'with delta K', delta_K)
    newK = update_vertex(mid_term(K_old))
    K = (1 - a) * newK + a * K
    delta_K = amax(abs(K - K_old))
    C += 1


plt.plot(times, K[0], label='0')
plt.plot(times, K[1], label='1')
plt.plot(times, K[2].imag, label='1i')
plt.plot(times, K[3].imag, label='0i')
plt.legend()
plt.show()
print(K[0, N // 2], K[1, N // 2], K[2, N // 2], K[3, N // 2])
