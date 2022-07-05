import matplotlib.pyplot as plt
from numpy import *
from functions import *


def ss_nca(v, eps, u, temperature, t_m, t_l, t_max, h, lamb):
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

    # numerical parameter for infinity
    N = 20 * t_max  # number of time data points
    ha = int(20 * h)
    start = linspace(0, t_max + h, N + ha + 1)
    times = linspace(0, t_max, N + 1)
    dt = start[1] - start[0]
    cutoff_factor = 100.0
    N_w = 5000000
    w = linspace(- 2 * t_l * cutoff_factor, 2 * t_l * cutoff_factor, N_w)
    dw = w[1] - w[0]
    d_dyson = 1e-6
    a = 0.3

    gam_w = gamma(w, epsilon_lead, t_l, t_m)

    delta_l_energy = [gam_w * f(w, miu[0], beta),
                      gam_w * f(w, miu[1], beta)]
    delta_g_energy = [gam_w * (1 - f(w, miu[0], beta)),
                      gam_w * (1 - f(w, miu[1], beta))]

    delta_l_temp = [ifftshift(fft(fftshift(delta_l_energy[0]))) * dw / pi,
                    ifftshift(fft(fftshift(delta_l_energy[1]))) * dw / pi]
    delta_g_temp = [ifftshift(fft(fftshift(delta_g_energy[0]))) * dw / pi,
                    ifftshift(fft(fftshift(delta_g_energy[1]))) * dw / pi]

    hl = zeros((3, N + ha + 1), complex)
    hg = zeros((3, N + ha + 1), complex)
    for i in range(N + ha + 1):
        hl[0][i] = conj(delta_l_temp[0][time_to_fftind(start[i], cutoff_factor, t_l, dw)])
        hl[1][i] = conj(delta_l_temp[1][time_to_fftind(start[i], cutoff_factor, t_l, dw)])
        hg[0][i] = delta_g_temp[0][time_to_fftind(start[i], cutoff_factor, t_l, dw)]
        hg[1][i] = delta_g_temp[1][time_to_fftind(start[i], cutoff_factor, t_l, dw)]
    hl[2] = hl[0] + hl[1]  # lesser hybridization function
    hg[2] = hg[0] + hg[1]  # greater

    # build the initial states

    G0 = zeros((4, N + ha + 1), complex)
    for t in range(N + ha + 1):
        for state_i in range(4):
            G0[state_i, t] = g(start[t], state_i, E) * exp(-5 * start[t] / t_max)
    G = copy(G0)
    SE = update_self_energy(G, hl, hg, N + ha)
    delta_G = d_dyson + 1
    # print("start iterations to calculate G (green function), SE (self energy)")
    C = 0
    while delta_G > d_dyson:
        G_old = copy(G)
        G = update_green(SE, G_old, G0, dt)
        SE = update_self_energy(G, hl, hg, N + ha)
        delta_G = amax(abs(G - G_old))
        C += 1
        print('G iteration number', C, 'with delta G', delta_G)
    print("NCA green function Converged within", delta_G, "after", C, "iterations.")

    hl[2] = hl[0] * exp(lamb / 2) + hl[1]  # lesser hybridization function
    hg[2] = hg[0] * exp(-lamb / 2) + hg[1]  # greater

    H_mat = zeros((4, 4, ha + 1, N + 1), complex)
    for t1 in range(ha + 1):
        for t2 in range(N + 1):
            H_mat[0, 1, t1, t2] = sign_time(hl[2], t1, t2)
            H_mat[0, 2, t1, t2] = sign_time(hl[2], t1, t2)
            H_mat[1, 0, t1, t2] = sign_time(hg[2], t1, t2)
            H_mat[1, 3, t1, t2] = sign_time(hl[2], t1, t2)
            H_mat[2, 0, t1, t2] = sign_time(hg[2], t1, t2)
            H_mat[2, 3, t1, t2] = sign_time(hl[2], t1, t2)
            H_mat[3, 1, t1, t2] = sign_time(hg[2], t1, t2)
            H_mat[3, 2, t1, t2] = sign_time(hg[2], t1, t2)
    K0 = zeros((4, 4, ha + 1, N + 1), complex)
    for j1 in range(ha + 1):
        for j2 in range(N + 1):
            for down in range(4):
                K0[down, down, j1, j2] = gen_bare_vertex(down, down, j1, j2, G[:, :N + 1])

    K = copy(K0)
    # print("start iterations to find the vertex function")

    for a in range(4):
        C = 0
        delta_K = d_dyson + 1
        while delta_K > d_dyson:
            K_old = copy(K[a])
            P = mult_vertex_single(K[a], N, H_mat, ha)
            K[a] = update_vertex_single(P, G[:, :N + 1], K0[a], N, dt, ha)
            delta_K = amax(abs(K[a] - K_old))
            C += 1
            print("...... calculating vertex .......")
            print(a, C, delta_K)
    return K, times, G, H_mat


def main():
    k, t, G, H = ss_nca(1, 0, 1, 1, 1, 1, 20, 10, 1)
    save("kss", k)
    Z = zeros(200, complex)
    for i in range(1, 200):
        Z[i] = find_fcs(k[:, :, :i, :], G, H[:, :, :i, :], t[1] - t[0])
    plt.plot(linspace(0, 1, 200), Z)
    plt.show()


if __name__ == "__main__":
    main()
