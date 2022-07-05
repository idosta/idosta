import matplotlib.pyplot as plt
from functions import *
import time


def NCA(v, eps, u, temperature, lamb, t_max, N, dim_l, t_m, t_l):
    ec = 2 * t_l
    beta = 1.0 / temperature
    V = v
    miu = array([V / 2, -V / 2])  # 0'th place for left and 1 for right lead
    U = u
    gate = eps
    epsilon0 = -U / 2 + gate
    E = (0, epsilon0, epsilon0, 2 * epsilon0 + U)
    epsilon_lead = 0
    # t_max is maximal time

    # numerical parameters
    # N is number of time points
    # N is number of time points
    n_cheb = 300  # number of coefficients
    N_lead = 1000  # number of sites in the lead
    N_w = 1000  # number of energy points
    dt = t_max / N
    times = linspace(0, t_max, N + 1)
    cutoff_factor = 1000
    dw = 0.01 * t_l
    w = arange(-ec * cutoff_factor, ec * cutoff_factor, dw)
    d_dyson = 1e-7
    slice = 6

    gam_w = gamma(w, epsilon_lead, t_l, t_m)

    delta_l_energy = [gam_w * f(w, beta, miu[0]), gam_w * f(w, beta, miu[1])]
    delta_g_energy = [gam_w * (1 - f(w, beta, miu[0])), gam_w * (1 - f(w, beta, miu[1]))]

    delta_l_temp = [ifftshift(fft(fftshift(delta_l_energy[0]))) * dw / pi,
                    ifftshift(fft(fftshift(delta_l_energy[1]))) * dw / pi]
    delta_g_temp = [ifftshift(fft(fftshift(delta_g_energy[0]))) * dw / pi,
                    ifftshift(fft(fftshift(delta_g_energy[1]))) * dw / pi]

    hl = zeros((3, N + 1), complex)
    hg = zeros((3, N + 1), complex)
    for i in range(N + 1):
        hl[0][i] = conj(delta_l_temp[0][time_to_fftind(times[i], cutoff_factor, t_l, dw)])
        hl[1][i] = conj(delta_l_temp[1][time_to_fftind(times[i], cutoff_factor, t_l, dw)])
        hg[0][i] = delta_g_temp[0][time_to_fftind(times[i], cutoff_factor, t_l, dw)]
        hg[1][i] = delta_g_temp[1][time_to_fftind(times[i], cutoff_factor, t_l, dw)]
    hl[2] = hl[0] + hl[1]
    hg[2] = hg[0] + hg[1]

    # build the initial states

    G0 = zeros((4, N + 1), complex)
    for t in range(N + 1):
        for state_i in range(4):
            G0[state_i, t] = g(times[t], state_i, E)
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
        print("...... calculating propagator .......")
        print(C, delta_G)

    # plt.plot(times, G[0].imag, label='0i')
    # plt.plot(times, G[1].imag, label='1i')
    # plt.plot(times, G[2], label='1')
    # plt.plot(times, G[3], label='0')
    # plt.legend()
    # plt.show()

    # if lamb == 0:
    #     for s in range(4):
    #         savetxt("G_ido" + str(s) + ".out",
    #                 c_[times, G[s, :].real, G[s, :].imag])
    print("NCA green function Converged within", d_dyson, "after", C, "iterations.")

    #  calculate the vertex function K

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
            for down in range(4):
                K0[down, down, j1, j2] = gen_bare_vertex(down, down, j1, j2, G)
    Ks = copy(K0[:, :, :N // slice + 1, :N // slice + 1])
    # print("start iterations to find the vertex function")

    for a in range(4):
        C = 0
        delta_K = d_dyson + 1
        while delta_K > d_dyson:
            K_old = copy(Ks[a])
            P = mult_vertex(Ks[a], N // slice, H_mat[:, :, :N // slice + 1, :N // slice + 1])
            Ks[a] = update_vertex(P, G[:, :N // slice + 1], K0[a, :, :N // slice + 1, :N // slice + 1], N // slice, dt)
            delta_K = amax(abs(Ks[a] - K_old))
            C += 1
            print("...... calculating vertex .......")
            print(a, C, delta_K)

    K = copy(K0)
    # print("start iterations to find the vertex function")

    K = guess_vertex(Ks, N // slice, K, N)

    for a in range(4):
        C = 0
        delta_K = d_dyson + 1
        while delta_K > d_dyson:
            K_old = copy(K[a])
            P = mult_vertex(K[a], N, H_mat)
            K[a] = update_vertex(P, G, K0[a], N, dt)
            delta_K = amax(abs(K[a] - K_old))
            C += 1
            print("...... calculating vertex .......")
            print(a, C, delta_K)

    print("NCA vertex function Converged within", d_dyson, "after", C, "iterations.", "for v, h = ", v, eps)
    # calculate partition function
    p0 = zeros(4)
    for i in range(4):
        p0[i] = exp(-E[i] * beta)
    P0 = p0 / sum(p0)
    save("Ku", K)
    save("P", P0)

    if lamb == 0:
        Pr = zeros((4, N), complex)
        for i in range(1):
            for tn in range(N):
                Pr[i, tn] = K[i, :, tn, tn] @ p0[:]
    Z = zeros(N, complex)
    for jt in range(N):
        temp_Z = 0
        for j in range(4):
            temp_Z += P0[:] @ K[j, :, jt, jt]
        Z[jt] = temp_Z
    return Z


def main():
    start_time = time.time()
    y = NCA(1, 0, 1, 1, 1, 20, 400, 1, 1, 1)
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.plot(linspace(0, 20, len(y)), log(y), label="0.1")
    plt.legend()
    plt.show()
    plt.plot(linspace(0, 20, len(y)), log(y).imag, label="0.1")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
