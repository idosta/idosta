import matplotlib.pyplot as plt
from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftshift, ifftshift
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
from numpy.polynomial.chebyshev import chebval


def find_c(v, eps, u, temperature, lamb, t_max, N, dim_l, t_m, t_l, dif):
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
        dt = t_max / (N - 1)
        times = linspace(0, t_max, N)
        cutoff_factor = 10
        dw = 0.01 * t_l
        w = arange(-ec * cutoff_factor, ec * cutoff_factor, dw)
        d_dyson = 1e-7

        def gen_hyb(dim, t_mol, t_lead, epsilon, NU, n_c, nw):
            # NU = number of sites
            Nd = int(NU ** (1 / dim))

            def build_h_3d(n, ep, tb):
                hamiltonian = sp.lil_matrix((n ** 3, n ** 3))
                for s in range(n ** 3):
                    hamiltonian[s, s] = ep
                    if s % Nd != 0:  # nearest in same row
                        hamiltonian[s - 1, s] = tb
                        hamiltonian[s, s - 1] = tb
                    if s - Nd + 1 > 0:  # previous row
                        hamiltonian[s, s - Nd] = tb
                        hamiltonian[s - Nd, s] = tb
                    if s - Nd ** 2 + 1 > 0:  # previous plane
                        hamiltonian[s, s - Nd ** 2] = tb
                        hamiltonian[s - Nd ** 2, s] = tb
                return hamiltonian.tocsr()

            def build_h_2d(n, ep, tb):
                hamiltonian = sp.lil_matrix((n ** 2, n ** 2))
                for s in range(n ** 2):
                    hamiltonian[s, s] = ep
                    if s % Nd != 0:  # nearest in same row
                        hamiltonian[s - 1, s] = tb
                        hamiltonian[s, s - 1] = tb
                    if s - Nd + 1 > 0:  # previous row
                        hamiltonian[s, s - Nd] = tb
                        hamiltonian[s - Nd, s] = tb
                return hamiltonian.tocsr()

            def build_h_1d(n, ep, tb):
                hamiltonian = sp.lil_matrix((n, n))
                for s in range(n):
                    hamiltonian[s, s] = ep
                for s in range(n - 1):
                    hamiltonian[s, s + 1] = tb
                    hamiltonian[s + 1, s] = tb
                return hamiltonian.tocsr()

            def build_h(n, ep, t_le):
                if dim == 1:
                    return build_h_1d(n, ep, t_le), dim
                if dim == 2:
                    return build_h_2d(n, ep, t_le), dim
                if dim == 3:
                    return build_h_3d(n, ep, t_le), dim

            H, d = build_h(Nd, epsilon, t_lead)
            E_max = float(eigsh(H, 1, which='LA', return_eigenvectors=False))
            E_min = float(eigsh(H, 1, which='SA', return_eigenvectors=False))
            d = (E_max - E_min) / 2
            b = (E_max + E_min) / 2
            H, d = build_h(Nd, (epsilon - b) / d, t_lead / d)

            c = zeros((3, Nd ** d))
            c[0, 0] = 1
            cz = copy(c[0, :])
            c[1] = H @ c[0]
            mu = zeros(n_c)
            mu[0] = 0.5
            mu[1] = c[0, :] @ c[1, :]
            for ic in range(2, n_c):
                c[2, :] = 2 * H @ c[1, :] - c[0, :]
                mu[ic] = cz @ c[2, :]
                c[0, :] = copy(c[1, :])
                c[1, :] = copy(c[2, :])
            w_sp = linspace(-1, 1, nw)
            D = chebval(w_sp, mu) * (2 / pi) / sqrt(1 - w_sp ** 2)
            w_sp = w_sp * d + b
            D = D / d
            D = D * pi * t_mol ** 2
            return w_sp, D

        # define mathematical functions
        def fft_integral(x, y):
            temp = (fftconvolve(x, y)[:N] - 0.5 * (x[:N] * y[0] + x[0] * y[:N])) * dt
            temp[N:] = 0
            return temp

        def integral_green(bare_green, self_energy, old_green):
            total23, total21 = fft_integral(self_energy, old_green), fft_integral(self_energy, bare_green)
            return 0.5 * (fft_integral(bare_green, total23) + fft_integral(old_green, total21))

        # def gamma(energy):
        #     return 0.5 * ga / ((1 + exp(nu * (energy - ec))) * (1 + exp(-nu * (energy + ec))))

        def gamma_c(energy):
            gam = gen_hyb(dim_l, t_m, t_l, 0, N_lead, n_cheb, N_w)
            y = zeros(len(energy))
            for en in range(len(energy)):
                if min(gam[0]) < energy[en] < max(gam[0]):
                    ind = argmin(abs(gam[0] - energy[en]))
                    y[en] = gam[1][ind]
            return y

        def f(energy, mu):
            return 1 / (1 + exp(beta * (energy - mu)))

        def gamma(w_sp):
            P = zeros(len(w_sp))
            for en in range(len(w_sp)):
                if abs(w_sp[en] - epsilon_lead) < (2 * t_l):
                    P[en] = (t_m ** 2 / (2 * t_l ** 2)) * sqrt(4 * t_l ** 2 - (w_sp[en] - epsilon_lead) ** 2)
            return P

        gam_w = gamma(w)

        delta_l_energy = [gam_w * f(w, miu[0]), gam_w * f(w, miu[1])]
        delta_g_energy = [gam_w * (1 - f(w, miu[0])), gam_w * (1 - f(w, miu[1]))]

        delta_l_temp = [ifftshift(fft(fftshift(delta_l_energy[0]))) * dw / pi,
                        ifftshift(fft(fftshift(delta_l_energy[1]))) * dw / pi]
        delta_g_temp = [ifftshift(fft(fftshift(delta_g_energy[0]))) * dw / pi,
                        ifftshift(fft(fftshift(delta_g_energy[1]))) * dw / pi]

        def time_to_fftind(ti):
            return int(cutoff_factor * ec / dw) + (round(ti * cutoff_factor * ec / pi))

        hl = zeros((3, N), complex)
        hg = zeros((3, N), complex)
        for i in range(N):
            hl[0][i] = conj(delta_l_temp[0][time_to_fftind(times[i])])
            hl[1][i] = conj(delta_l_temp[1][time_to_fftind(times[i])])
            hg[0][i] = delta_g_temp[0][time_to_fftind(times[i])]
            hg[1][i] = delta_g_temp[1][time_to_fftind(times[i])]
        hl[2] = hl[0] + hl[1]
        hg[2] = hg[0] + hg[1]

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

        # plt.plot(times, G[0].imag, label='0i')
        # plt.plot(times, G[1].imag, label='1i')
        # plt.plot(times, G[2], label='1')
        # plt.plot(times, G[3], label='0')
        # plt.legend()
        # plt.show()

        # calculating v

        #    print(C, delta_G)
        # if lamb == 0:
        #     for s in range(4):
        #         savetxt("G_ido" + str(s) + ".out",
        #                 c_[times, G[s, :].real, G[s, :].imag])
        # print("NCA green function Converged within", d_dyson, "after", C, "iterations.")

        #  calculate the vertex function K

        def sign_time(fun, t_i, t_f):
            if t_f - t_i >= 0:
                return fun[t_f - t_i]
            else:
                return conj(fun[t_i - t_f])

        # savetxt("/home/ido/NCA/temp_results/Delta_lesserI.out", c_[times, imag(hl), -real(hl)])
        # savetxt("/home/ido/NCA/temp_results/Delta_greaterI.out", c_[times, imag(hg), -real(hg)])
        # FCS time
        for i in range(N):
            hl[2][i] = hl[0][i] * exp(lamb / 2) + hl[1][i]
            hg[2][i] = hg[0][i] * exp(-lamb / 2) + hg[1][i]
        H_mat = zeros((4, 4, N, N), complex)
        for t1 in range(N):
            for t2 in range(N):
                H_mat[0, 1, t1, t2] = sign_time(hl[2], t1, t2)
                H_mat[0, 2, t1, t2] = sign_time(hl[2], t1, t2)
                H_mat[1, 0, t1, t2] = sign_time(hg[2], t1, t2)
                H_mat[1, 3, t1, t2] = sign_time(hl[2], t1, t2)
                H_mat[2, 0, t1, t2] = sign_time(hg[2], t1, t2)
                H_mat[2, 3, t1, t2] = sign_time(hl[2], t1, t2)
                H_mat[3, 1, t1, t2] = sign_time(hg[2], t1, t2)
                H_mat[3, 2, t1, t2] = sign_time(hg[2], t1, t2)

        def gen_bare_vertex(final_state, initial_state, final_time, initial_time, green_function):
            return conj(green_function[final_state, final_time]) * green_function[initial_state, initial_time]

        def multi_vertex(k):
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
                P = multi_vertex(K[a])
                K[a] = update_vertex(P, G, K0[a])
                delta_K = amax(abs(K[a] - K_old))
                C += 1
                # print(C, delta_K)

        print("NCA vertex function Converged within", d_dyson, "after", C, "iterations.", "for v, h = ", v, eps)
        # calculate partition function
        p0 = zeros(4)
        for i in range(4):
            p0[i] = exp(-E[i] * beta)
        p0 = p0 / sum(p0)

        if lamb == 0:
            Pr = zeros((4, N), complex)
            for i in range(4):
                for tn in range(N):
                    Pr[i, tn] = K[i, :, tn, tn] @ p0[:]
        Z = zeros(N, complex)
        for jt in range(N // 4):
            temp_Z = 0
            for i in range(4):
                for j in range(4):
                    temp_Z += p0[i] * K[j, i, 0, jt]
            Z[jt] = temp_Z
        return Z, K, times, G, p0, hl, hg

    v, eps, u, temperature, lamb, t_max, N, dim_l, t_m, t_l = 1, 0, 1, 0.1, 0, 20, 200, 1, 1, 1

    nca = NCA(v, eps, u, temperature, lamb, t_max, N, dim_l, t_m, t_l)
    K, times, p0 = nca[1], nca[2], nca[4]

    z_diff = zeros((dif // 2 + 1, N // 2), complex)
    for o in range(dif // 2 + 1):
        for jt in range(o * N // dif, o * N // dif + N // 2):
            temp_Z = 0
            for i in range(4):
                for j in range(4):
                    temp_Z += p0[i] * K[j, i, o * N // dif, jt]
            z_diff[o, jt - o * N // dif] = temp_Z
    c = log(z_diff[dif // 2, -1] / z_diff[dif // 2 - 1, -1]) / (times[N // dif] - times[0])
    return c, nca

