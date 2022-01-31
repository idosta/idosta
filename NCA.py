from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftshift, ifftshift


# dot state description (0, down, up, 1)
# physical parameters
def NCA(v, eps, temperature, lamb, t_max, N):
    ga = 1.0
    ec = 50.0 * ga
    nu = 1.0 / ga
    beta = 1.0 / (ga * temperature)
    V = v * ga
    miu = array([V / 2, -V / 2])  # 0'th place for left and 1 for right lead
    U = 5 * ga
    gate = eps * ga
    epsilon0 = -U / 2 + gate
    E = (0, epsilon0, epsilon0, 2 * epsilon0 + U)
    # t_max is maximal time

    # numerical parameters
    # N is number of time points
    dt = t_max / (N - 1)
    times = linspace(0, t_max, N)
    cutoff_factor = 100.0
    dw = 0.01
    w = arange(-ec * cutoff_factor, ec * cutoff_factor, dw)
    d_dyson = 1e-10

    # define mathematical functions
    def fft_integral(x, y):
        temp = (fftconvolve(x, y)[:N] - 0.5 * (x[:N] * y[0] + x[0] * y[:N])) * dt
        temp[N:] = 0
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
        return int(cutoff_factor * ec / dw) + (round(ti * cutoff_factor * ec / pi))

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
    for t in range(N):
        for state_i in range(4):
            G0[state_i, t] = g(times[t], state_i)
    G = copy(G0)
    SE = update_self_energy(N, G)
    delta_G = d_dyson + 1
    # print("start iterations to calculate G (green function), SE (self energy)")
    C = 0
    while delta_G > d_dyson:
        G_old = copy(G)
        G = update_green(SE, G_old, G0)
        SE = update_self_energy(N, G)
        delta_G = amax(abs(G - G_old))
        C += 1

    #    print(C, delta_G)
    if lamb == 0:
        for s in range(4):
            savetxt("/home/ido/NCA/temp_results/G_ido" + str(s) + ".out",
                    c_[times, G[s, :].real, G[s, :].imag])
        print("NCA green function Converged within", d_dyson, "after", C, "iterations.")

    #  calculate the vertex function K

    def sign_time(fun, t_i, t_f):
        if t_f - t_i >= 0:
            return fun[t_f - t_i]
        else:
            return conj(fun[t_i - t_f])

    savetxt("/home/ido/NCA/temp_results/Delta_lesserI.out", c_[times, imag(hl), -real(hl)])
    savetxt("/home/ido/NCA/temp_results/Delta_greaterI.out", c_[times, imag(hg), -real(hg)])

    for i in range(N):
        hl[i] = hl[i] * exp(-1j * lamb * times[i])
        hg[i] = hg[i] * exp(1j * lamb * times[i])  # check for problems with lambda
    H_mat = zeros((4, 4, N, N), complex)
    for t1 in range(N):
        for t2 in range(N):
            H_mat[0, 1, t1, t2] = sign_time(hl, t1, t2)
            H_mat[0, 2, t1, t2] = sign_time(hl, t1, t2)
            H_mat[1, 0, t1, t2] = sign_time(hg, t1, t2)
            H_mat[1, 3, t1, t2] = sign_time(hl, t1, t2)
            H_mat[2, 0, t1, t2] = sign_time(hg, t1, t2)
            H_mat[2, 3, t1, t2] = sign_time(hl, t1, t2)
            H_mat[3, 1, t1, t2] = sign_time(hg, t1, t2)
            H_mat[3, 2, t1, t2] = sign_time(hg, t1, t2)

    def gen_bare_vertex(final_state, initial_state, final_time, initial_time, green_function):
        return conj(green_function[final_state, final_time]) * green_function[initial_state, initial_time]

    def mult_vertex(k):
        va = zeros((4, N, N), complex)
        for q in range(4):
            for m in range(4):
                va[q] += k[m] * H_mat[m, q]
        return va

    def update_vertex(p, gr, k0):
        temp = copy(k0)
        for i_f in range(4):
            c = zeros((N, N), complex)
            for t_2 in range(N):
                c[:, t_2] = fft_integral(p[i_f, :, t_2], conj(gr[i_f, :]))
            for t_1 in range(N):
                temp[i_f, t_1, :] += fft_integral((gr[i_f, :]), c[t_1, :])
        return temp

    K0 = zeros((4, 4, N, N), complex)
    for j1 in range(N):
        for j2 in range(N):
            for down in range(4):
                K0[down, down, j1, j2] = gen_bare_vertex(down, down, j1, j2, G)
    K = copy(K0)
    # print("start iterations to find the vertex function")

    for a in range(4):
        C = 0
        delta_K = d_dyson + 1
        while delta_K > d_dyson:
            K_old = copy(K[a])
            P = mult_vertex(K[a])
            K[a] = update_vertex(P, G, K0[a])
            delta_K = amax(abs(K[a] - K_old))
            C += 1
            # print(a, C, delta_K)

    print("NCA vertex function Converged within", d_dyson, "after", C, "iterations.")
    if lamb == 0:
        P = zeros((4, 4, N), complex)
        Z = zeros(N, complex)
        for i in range(4):
            for j in range(4):
                for tn in range(N):
                    P[i, j, tn] = K[i, j, tn, tn]
                    savetxt("/home/ido/NCA/temp_results/P_ido" + str(i) + str(j) + ".out", c_[times, P[i, j, :].real])

    # calculate partition function
    p0 = zeros(4)
    for i in range(4):
        p0[i] = exp(-beta * E[i])
    p0 = p0 / sum(p0)
    Z = zeros(N, complex)
    for jt in range(N):
        temp_Z = 0
        for i in range(4):
            for j in range(4):
                temp_Z += p0[i] * K[j, i, jt, jt]
        Z[jt] = temp_Z
    return Z


NCA(10, 0, 1, 0, 5, 201)

# def clog(z, axis=0):
#     logz = log(z)
#     for ilog in range(len(logz) - 1):
#         if abs(logz[ilog] - logz[ilog + 1]) > abs(logz[ilog] - logz[ilog + 1] - 2j * pi):
#             logz[ilog + 1:] += 2j * np.pi
#         elif abs(logz[ilog] - logz[ilog + 1]) > abs(logz[ilog] - logz[ilog + 1] + 2j * pi):
#             logz[ilog + 1:] -= 2j * pi
#     return logz
#
#
# def c_cal(dx, v, eps, temperature, t_max, N):
#     def g(lambd):  # calculate the partition function log at lambda
#         return clog(NCA(v, eps, temperature, lambd, t_max, N))
#
#     return -1j * (g(dx) - g(-dx)) / (2 * dx)
#
#
# N = 201
# t_max = 5
# C = c_cal(0.00001, 5, 5, 1, t_max, N)
# times = linspace(0, t_max, N)
# I = zeros(N, complex)
# for i in range(1, N - 1):
#     I[i] = (C[i + 1] - C[i - 1]) / (times[i + 1] - times[i - 1])
# I[N - 1] = I[N - 2]
# I[0] = I[1]
# savetxt("/home/ido/NCA/temp_results/0.out", c_[times, C.real, I.real])
