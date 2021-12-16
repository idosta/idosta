import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from numpy import linalg as LA


def Z_NCA(lamb):
    # dot state description (0, down, up, 1)
    # physical parameters
    from numpy import ndarray

    ga = 1
    ec = 50 * ga
    nu = 1 / ga
    beta = 1 / ga
    V = 20 * ga
    miu = array([-V / 2, V / 2])  # 0'th place for left and 1 for right lead
    U = 40 * ga
    gate = 0
    epsilon0 = -U / 2 + gate * ga
    E = (0, epsilon0, epsilon0, 2 * epsilon0 + U)
    lamb = 0
    t_max = 5  # maximal time

    # numerical parameters
    Nx = 1000  # number of points for hybridization integral
    nec = 3  # limit for the hyb integral function
    d_dyson = 0.00001
    N = 20  # number of time points
    dt = t_max / (N-1)
    times = linspace(0, t_max, N)


    # define mathematical functions
    def integral_green(bare_green, old_green, self_energy, final_time):
        total = 0
        for x in range(final_time):
            for inner_time in range(x):
                total += bare_green[final_time - x] * self_energy[x - inner_time] * old_green[inner_time] * dt**2
        return total


    def gamma(energy):
        return 0.5 * ga / ((1 + exp(nu * (energy - ec))) * (1 + exp(-nu * (energy + ec))))


    def f(energy, mu):
        return 1 / (1 + exp(beta * (energy - mu)))


    def hyb_lesser(mu):
        temp = zeros(N, complex)
        for jt in range(N):
            x = linspace(-nec, nec, Nx)
            y = exp(1j * x * times[jt]) * gamma(x) * f(x, mu) / np.pi
            temp[jt] = trapz(y, x)
        return temp


    hl = array([hyb_lesser(miu[0]), hyb_lesser(miu[1])])


    def hyb_greater(mu):
        temp = zeros(N, complex)
        for jt in range(N):
            x = linspace(-nec, nec, Nx)
            y = exp(-1j * x * jt) * gamma(x) * (1 - f(x, mu)) / np.pi
            temp[jt] = trapz(y, x)
        return temp


    hg = array([hyb_greater(miu[0]), hyb_greater(miu[1])])


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


    def cross_branch_hyb(down, up, jt):
        temp = 0
        for spin in [0, 1]:
            for lead in [0, 1]:
                temp = temp + (
                        hl[lead, jt] * d_op(spin, 0, down, up) * d_op(spin, 1, up, down)
                        * exp(-1j * lamb * times[jt]) + hg[lead, jt] * d_op(spin, 1, down, up) *
                        d_op(spin, 0, up, down)) * exp(1j * lamb * times[jt])
        return temp


    CBH = zeros((4, 4, N), complex)
    for down in range(3):
        for up in range(3):
            for it in range(N):
                CBH[down, up, it] = cross_branch_hyb(down, up, it)


    def g(time, site):  # t for time and j for the site number in the dot
        return exp(-1j * E[site] * time)


    def update_green(number_of_times, self_energy, old_green, bare_green):
        temp = copy(bare_green)
        for jt in range(number_of_times):
            for site in range(3):
                temp[site, jt] -= integral_green(bare_green[site, :], old_green[site, :], self_energy[site, :], jt)
        return temp


    def update_self_energy(number_of_times, green):
        temp = zeros((4, number_of_times), complex)
        for site in range(3):
            for jt in range(number_of_times):
                for spin in [0, 1]:
                    for lead in [0, 1]:
                        for beta_site in range(3):
                            temp[site, jt] += (hl[lead, jt] * d_op(spin, 0, site, beta_site) *
                                               d_op(spin, 1, beta_site, site) + hg[lead, jt] *
                                               d_op(spin, 1, site, beta_site) *
                                               d_op(spin, 0, beta_site, site)) * green[beta_site, jt]
        return temp


    # build the initial states


    G0 = zeros((4, N), complex)
    SE = zeros((4, N), complex)
    for t in range(N):
        for down in range(3):
            G0[down, t] = g(times[t], down)

    G = copy(G0)
    SE = update_self_energy(N, G)
    delta_G = d_dyson + 1
    # start iterations to calculate G (green function), SE (self energy)
    while delta_G > d_dyson:
        G_old = copy(G)
        G = update_green(N, SE, G_old, G0)
        SE = update_self_energy(N, G)
        delta_G = LA.norm(G - G_old)
        print(delta_G)


    #  calculate the vertex function K



    # dot state description (0, down, up, 1)
    # physical parameters
    from numpy import ndarray

    ga = 1
    ec = 50 * ga
    nu = 1 / ga
    beta = 1 / ga
    V = 20 * ga
    miu = array([-V / 2, V / 2])  # 0'th place for left and 1 for right lead
    U = 40 * ga
    gate = 0
    epsilon0 = -U / 2 + gate * ga
    E = (0, epsilon0, epsilon0, 2 * epsilon0 + U)
    lamb = 0
    t_max = 5  # maximal time

    # numerical parameters
    Nx = 1000  # number of points for hybridization integral
    nec = 3  # limit for the hyb integral function
    d_dyson = 0.001
    N = 100  # number of time points
    dt = t_max / (N-1)
    times = linspace(0, t_max, N)


    # define mathematical functions
    def integral_green(bare_green, old_green, self_energy, final_time):
        total = 0
        for x in range(final_time):
            for inner_time in range(x):
                total += bare_green[final_time - x] * self_energy[x - inner_time] * old_green[inner_time] * dt**2
        return total


    def gamma(energy):
        return 0.5 * ga / ((1 + exp(nu * (energy - ec))) * (1 + exp(-nu * (energy + ec))))


    def f(energy, mu):
        return 1 / (1 + exp(beta * (energy - mu)))


    def hyb_lesser(mu):
        temp = zeros(N, complex)
        for jt in range(N):
            x = linspace(-nec, nec, Nx)
            y = exp(1j * x * times[jt]) * gamma(x) * f(x, mu) / np.pi
            temp[jt] = trapz(y, x)
        return temp


    hl = array([hyb_lesser(miu[0]), hyb_lesser(miu[1])])


    def hyb_greater(mu):
        temp = zeros(N, complex)
        for jt in range(N):
            x = linspace(-nec, nec, Nx)
            y = exp(-1j * x * jt) * gamma(x) * (1 - f(x, mu)) / np.pi
            temp[jt] = trapz(y, x)
        return temp


    hg = array([hyb_greater(miu[0]), hyb_greater(miu[1])])


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


    def cross_branch_hyb(down, up, jt):
        temp = 0
        for spin in [0, 1]:
            for lead in [0, 1]:
                temp = temp + (
                        hl[lead, jt] * d_op(spin, 0, down, up) * d_op(spin, 1, up, down)
                        * exp(-1j * lamb * times[jt]) + hg[lead, jt] * d_op(spin, 1, down, up) *
                        d_op(spin, 0, up, down)) * exp(1j * lamb * times[jt])
        return temp


    CBH = zeros((4, 4, N), complex)
    for down in range(3):
        for up in range(3):
            for it in range(N):
                CBH[down, up, it] = cross_branch_hyb(down, up, it)


    def g(time, site):  # t for time and j for the site number in the dot
        return exp(-1j * E[site] * time)


    def update_green(number_of_times, self_energy, old_green, bare_green):
        temp = copy(bare_green)
        for jt in range(number_of_times):
            for site in range(3):
                temp[site, jt] -= integral_green(bare_green[site, :], old_green[site, :], self_energy[site, :], jt)
        return temp


    def update_self_energy(number_of_times, green):
        temp = zeros((4, number_of_times), complex)
        for site in range(3):
            for jt in range(number_of_times):
                for spin in [0, 1]:
                    for lead in [0, 1]:
                        for beta_site in range(3):
                            temp[site, jt] += (hl[lead, jt] * d_op(spin, 0, site, beta_site) *
                                               d_op(spin, 1, beta_site, site) + hg[lead, jt] *
                                               d_op(spin, 1, site, beta_site) *
                                               d_op(spin, 0, beta_site, site)) * green[beta_site, jt]
        return temp


    # build the initial states


    G0 = zeros((4, N), complex)
    SE = zeros((4, N), complex)
    for t in range(N):
        for down in range(3):
            G0[down, t] = g(times[t], down)

    G = copy(G0)
    SE = update_self_energy(N, G)
    delta_G = d_dyson + 1
    # start iterations to calculate G (green function), SE (self energy)
    while delta_G > d_dyson:
        G_old = copy(G)
        G = update_green(N, SE, G_old, G0)
        SE = update_self_energy(N, G)
        delta_G = LA.norm(G - G_old)
        print(delta_G)


    #  calculate the vertex function K


    def gen_bare_vertex(final_state, initial_state, final_time, initial_time, green_function):
        return conj(green_function[final_state, final_time]) * green_function[initial_state, initial_time] \
               * int(bool(initial_state == final_state))


    def integrate_vertex(t1, t2, b, bt, a, at, bare_vertex, vertex):
        sum_int = 0
        for r in range(t1):
            for s in range(t2):
                sum_int += bare_vertex[bt, b, t1 - r, t2 - s] * CBH[at, bt, r - s] * vertex[a, at, r, s] * dt ** 2
        return sum_int


    def update_vertex(old_vertex, bare_vertex):
        temp = copy(bare_vertex)
        for ti in range(N):
            for tf in range(N):
                for a in range(3):
                    for b in range(3):
                        for at in range(3):
                            for bt in range(3):
                                temp[a, b, ti, tf] += integrate_vertex(ti, tf, b, bt, a, at, bare_vertex, old_vertex)
        return temp


    K0 = zeros((4, 4, N, N), complex)
    for j1 in range(N):
        for j2 in range(N):
            for down in range(3):
                K0[down, down, j1, j2] = gen_bare_vertex(down, down, j1, j2, G)

    K = copy(K0)
    delta_K = d_dyson + 1
    while delta_K > d_dyson:
        K_old = copy(K)
        K = update_vertex(K_old, K0)
        delta_K = LA.norm(K_old - K)
        print(delta_K)

    # calculate partition function

    Z = []
    for jt in range(N):
        temp = 0
        for i in range(3):
            for j in range(3):
                temp += G0[i] * K[j, i, jt, jt]
        Z.append(temp)

    def gen_bare_vertex(final_state, initial_state, final_time, initial_time, green_function):
        return conj(green_function[final_state, final_time]) * green_function[initial_state, initial_time] \
               * int(bool(initial_state == final_state))


    def integrate_vertex(t1, t2, b, bt, a, at, bare_vertex, vertex):
        sum_int = 0
        for r in range(t1):
            for s in range(t2):
                sum_int += bare_vertex[bt, b, t1 - r, t2 - s] * CBH[at, bt, r - s] * vertex[a, at, r, s] * dt ** 2
        return sum_int


    def update_vertex(old_vertex, bare_vertex):
        temp = copy(bare_vertex)
        for ti in range(N):
            for tf in range(N):
                for a in range(3):
                    for b in range(3):
                        for at in range(3):
                            for bt in range(3):
                                temp[a, b, ti, tf] += integrate_vertex(ti, tf, b, bt, a, at, bare_vertex, old_vertex)
        return temp


    K0 = zeros((4, 4, N, N), complex)
    for j1 in range(N):
        for j2 in range(N):
            for down in range(3):
                K0[down, down, j1, j2] = gen_bare_vertex(down, down, j1, j2, G)

    K = copy(K0)
    delta_K = d_dyson + 1
    while delta_K > d_dyson:
        K_old = copy(K)
        K = update_vertex(K_old, K0)
        delta_K = LA.norm(K_old - K)
        print(delta_K)

    # calculate partition function

    Z = []
    for jt in range(N):
        temp = 0
        for i in range(3):
            for j in range(3):
                temp += G0[i] * K[j, i, jt, jt]
        Z.append(temp)
    return Z

