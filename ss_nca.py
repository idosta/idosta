import matplotlib.pyplot as plt
from numpy import *

from functions import *


def main():
    v, eps, u, temperature, t_m, t_l = 1, 0, 1, 1, 1, 1
    lamb = 0
    ga = (t_m ** 2) / t_l
    epsilon_lead = t_l * ga  # adding elctron energy on the lead
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

    t_max = 30  # numerical parameter for infinity
    N = 20 * t_max  # number of time data points
    times = linspace(-t_max / 2, t_max / 2, N + 1)
    times_plus = linspace(0, t_max, N + 1)
    dt = times[1] - times[0]
    cutoff_factor = 100.0
    N_w = 5000000
    w = linspace(- 2 * t_l * cutoff_factor, 2 * t_l * cutoff_factor, N_w)
    dw = w[1] - w[0]
    d_dyson = 1e-6
    a = 0.3

    delta_l_energy = [gamma(w, epsilon_lead, t_l, t_m) * f(w, miu[0], beta),
                      gamma(w, epsilon_lead, t_l, t_m) * f(w, miu[1], beta)]
    delta_g_energy = [gamma(-w, epsilon_lead, t_l, t_m) * (1 - f(w, miu[0], beta)),
                      gamma(-w, epsilon_lead, t_l, t_m) * (1 - f(w, miu[1], beta))]

    delta_l_temp = [ifftshift(fft(fftshift(delta_l_energy[0]))) * dw / pi,
                    ifftshift(fft(fftshift(delta_l_energy[1]))) * dw / pi]
    delta_g_temp = [ifftshift(fft(fftshift(delta_g_energy[0]))) * dw / pi,
                    ifftshift(fft(fftshift(delta_g_energy[1]))) * dw / pi]

    hl = zeros((3, N + 1), complex)
    hg = zeros((3, N + 1), complex)
    for i in range(N + 1):
        hl[0][i] = conj(delta_l_temp[0][time_to_fftind(times_plus[i], cutoff_factor, t_l, dw)])
        hl[1][i] = conj(delta_l_temp[1][time_to_fftind(times_plus[i], cutoff_factor, t_l, dw)])
        hg[0][i] = delta_g_temp[0][time_to_fftind(times_plus[i], cutoff_factor, t_l, dw)]
        hg[1][i] = delta_g_temp[1][time_to_fftind(times_plus[i], cutoff_factor, t_l, dw)]
    hl[2] = hl[0] + hl[1]  # lesser hybridization function
    hg[2] = hg[0] + hg[1]  # greater

    # build the initial states

    G0 = zeros((4, N + 1), complex)
    for t in range(N + 1):
        for state_i in range(4):
            G0[state_i, t] = g(times_plus[t], state_i, E) * exp(-5 * times_plus[t] / t_max)
    G = copy(G0)
    SE = update_self_energy(G, hl, hg, N)
    delta_G = d_dyson + 1
    # print("start iterations to calculate G (green function), SE (self energy)")
    C = 0
    while delta_G > d_dyson:
        G_old = copy(G)
        G = update_green(SE, G_old, G0, dt)
        SE = update_self_energy(G, hl, hg, N)
        delta_G = amax(abs(G - G_old))
        C += 1
        print('G iteration number', C, 'with delta G', delta_G)
    print("NCA green function Converged within", delta_G, "after", C, "iterations.")

    hl[2] = hl[0] * exp(lamb / 2) + hl[1]  # lesser hybridization function
    hg[2] = hg[0] * exp(-lamb / 2) + hg[1]  # greater

    # calculating vertex function
    hL = conj_maker(hl[2], N)
    hG = conj_maker(hg[2], N)

    K = zeros((4, N + 1), complex)
    for i in range(4):
        K[i] = conj_maker(G[i], N)
    K = normal_k(K, N)

    delta_K = d_dyson + 1
    C = 0
    It = zeros((4, N + 1), complex)
    while delta_K > d_dyson:
        K_old = copy(K)
        It = mid_term(K_old, hL, hG, N)
        print('K iteration number', C, 'with delta K', delta_K)
        newK = update_vertex_ss(It, G, dt, N)
        K = (1 - a) * normal_k(newK, N) + a * K
        delta_K = amax(abs(K - K_old))
        C += 1

    print("NCA vertex function Converged within", delta_K, "after", C, "iterations.")

    plt.plot(times, K[0], label='0')
    plt.plot(times, K[1], label='1')
    plt.plot(times, K[2].imag, label='1i')
    plt.plot(times, K[3].imag, label='0i')
    plt.legend()
    plt.show()

    save("Kss", K)


if __name__ == "__main__":
    main()
