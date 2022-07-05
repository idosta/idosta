from numpy import *
from scipy.signal import fftconvolve
from scipy.fftpack import fft, fftshift, ifftshift
from scipy.sparse.linalg import eigsh
import scipy.sparse as sp
from numpy.polynomial.chebyshev import chebval


def fft_integral(x, y, dt):
    temp = (fftconvolve(x, y)[0:len(x)] - 0.5 * (x[0:len(x)] * y[0] + x[0] * y[0:len(x)])) * dt
    return temp


def integral_green(bare_green, self_energy, old_green, dt):
    total23, total21 = fft_integral(self_energy, old_green, dt), fft_integral(self_energy, bare_green, dt)
    return 0.5 * (fft_integral(bare_green, total23, dt) + fft_integral(old_green, total21, dt))


def f(energy, mu, beta):
    return 1 / (1 + exp(beta * (energy - mu)))


def time_to_fftind(ti, cutoff_factor, t_l, dw):
    return int(cutoff_factor * 2 * t_l / dw) + (round(ti * cutoff_factor * 2 * t_l / pi))


def g(time, site, E):  # t for time and j for the site number in the dot
    return exp(-1j * E[site] * time)


def update_green(self_energy, old_green, bare_green, dt):
    temp = copy(bare_green)
    for site in range(4):
        temp[site, :] -= integral_green(bare_green[site, :], self_energy[site, :], old_green[site, :], dt)
    return temp


def update_self_energy(green, Hl, Hg, N):
    temp = zeros((4, N + 1), complex)
    for t_se in range(N + 1):
        temp[0, t_se] = Hl[2][t_se] * (green[1, t_se] + green[2, t_se])
        temp[1, t_se] = Hg[2][t_se] * green[0, t_se] + Hl[2][t_se] * green[3, t_se]
        temp[2, t_se] = temp[1, t_se]
        temp[3, t_se] = Hg[2][t_se] * (green[1, t_se] + green[2, t_se])
    return temp


def conj_maker(A, N):
    hA = A[:N // 2 + 1]
    F = zeros(N + 1, complex)
    F[N // 2:] = copy(hA)
    F[:N // 2 + 1] = copy(conj(hA[::-1]))
    return F


def normal_k(vertex, N):
    vertex[:, :] = vertex[:, :] / sum(vertex[:, N // 2])
    return vertex


def mid_term(vertex, Hl, Hg, N):
    # this term is the multiplication of the hybridization function and the vertex function
    temp_mat = zeros((4, N + 1), complex)
    temp_mat[0] = (vertex[1] + vertex[2]) * Hl
    temp_mat[3] = (vertex[1] + vertex[2]) * Hg
    temp_mat[1] = vertex[0] * Hg + vertex[3] * Hl
    temp_mat[2] = temp_mat[1]
    return temp_mat


def update_vertex_ss(M, G, dt, N):
    A = zeros((4, N + 1), complex)
    B = zeros((4, N + 1), complex)
    MR = M[:, ::-1]
    for c in range(4):
        A[c] = fft_integral(G[c, :], MR[c, :], dt)
        AR = A[:, ::-1].copy()
        B[c] = fft_integral(conj(G[c, :]), AR[c, :], dt)
    return B


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


def gamma_c(energy):
    gam = gen_hyb(dim_l, t_m, t_l, 0, N_lead, n_cheb, N_w)
    y = zeros(len(energy))
    for en in range(len(energy)):
        if min(gam[0]) < energy[en] < max(gam[0]):
            ind = argmin(abs(gam[0] - energy[en]))
            y[en] = gam[1][ind]
    return y


def f(energy, mu, beta):
    return 1 / (1 + exp(beta * (energy - mu)))


def gamma(w_sp, epsilon_lead, t_l, t_m):
    P = zeros(len(w_sp))
    for en in range(len(w_sp)):
        if abs(w_sp[en] - epsilon_lead) < (2 * t_l):
            P[en] = (t_m ** 2 / (2 * t_l ** 2)) * sqrt(4 * t_l ** 2 - (w_sp[en] - epsilon_lead) ** 2)
    return P


def sign_time(fun, t_i, t_f):
    if t_f - t_i >= 0:
        return fun[t_f - t_i]
    else:
        return conj(fun[t_i - t_f])


def gen_bare_vertex(final_state, initial_state, final_time, initial_time, green_function):
    return conj(green_function[final_state, final_time]) * green_function[initial_state, initial_time]


def mult_vertex(k, N, H_mat):
    va = zeros((4, N + 1, N + 1), complex)
    for q in range(4):
        for m in range(4):
            va[q] += k[m] * H_mat[m, q]
    return va


def mult_vertex_single(k, N, H_mat, h):
    va = zeros((4, h + 1, N + 1), complex)
    for q in range(4):
        for m in range(4):
            va[q] += k[m] * H_mat[m, q]
    return va


def update_vertex(p, gr, k0, N, dt):
    temp = copy(k0)
    for i_f in range(4):
        c = zeros((N + 1, N + 1), complex)
        for t_2 in range(N + 1):
            c[:, t_2] = fft_integral(p[i_f, :, t_2], conj(gr[i_f, :]), dt)
        for t_1 in range(N + 1):
            temp[i_f, t_1, :] += fft_integral((gr[i_f, :]), c[t_1, :], dt)
    return temp


def update_vertex_single(p, gr, k0, N, dt, h):
    temp = copy(k0)
    for i_f in range(4):
        c = zeros((h + 1, N + 1), complex)
        for t_2 in range(N + 1):
            c[:, t_2] = fft_integral(p[i_f, :, t_2], conj(gr[i_f, :]), dt)
        for t_1 in range(h + 1):
            temp[i_f, t_1, :] += fft_integral((gr[i_f, :]), c[t_1, :], dt)
    return temp


def update_vertex_fcs_ss(gr, K, H, N, dt):
    temp = zeros((4, 4, N + 1, N + 1), complex)
    for down in range(4):
        for up in range(4):
            temp[down, up] += K[down, up, 0, 0]
    for down in range(4):
        va = zeros((4, N + 1, N + 1), complex)
        for q in range(4):
            for up in range(4):
                va[q] += K[down, up] * H[up, q]
        for up in range(4):
            c = zeros((N + 1, N + 1), complex)
            for t_2 in range(N + 1):
                c[:, t_2] = fft_integral(va[up, :, t_2], conj(gr[up, :]), dt)
            for t_1 in range(N + 1):
                temp[down, up, t_1, :] += fft_integral((gr[up, :]), c[t_1, :], dt)
    return temp


def guess_vertex(Ks, Ns, Kl, Nl):
    # dt must be equal in the two vertex functions
    C = zeros((4, 4), complex)
    for b in range(4):
        for t in range(4):
            C[b, t] = (2 / Ns) * log(Ks[b, t, Ns, Ns] / Ks[b, t, Ns // 2, Ns // 2])
    for i in range(Ns):
        for j in range(Ns):
            Kl[:, :, i, j] = Ks[:, :, i, j]
    for d in range(Nl // Ns):
        for i in range(Ns):
            for j in range(Ns):
                for b in range(4):
                    for t in range(4):
                        Kl[b, t, Ns * d + i, Ns * d + j] = Ks[b, t, i, j] * exp(Ns * d * C[b, t])
    return Kl


def find_fcs(K, G, H, dt):
    N = len(K[0, 0, 0, :]) - 1
    h = len(K[0, 0, :, 0]) - 1

    def int_cal(a, b):
        va = zeros((h + 1, N + 1), complex)
        for up in range(4):
            va += K[a, up] * H[up, b]
        s = 0
        for ti in range(h):
            for tj in range(N):
                s += 2 * (dt ** 2) * real(conj(G[b, N + h - ti]) * G[b, N + h - tj] * va[ti, tj])
        return s

    fs = zeros(2, complex)
    for i in range(4):
        for j in range(4):
            fs[0] += G[i, h + N] * conj(G[j, h + N]) + int_cal(i, j)
            fs[1] += G[i, N] * conj(G[j, N])
    return log(fs[0] / fs[1]) / (h * dt)
