from numpy import *
import matplotlib.pyplot as plt
from numpy import linalg as LA

# dot state description (0, down, up, 1)
# physical parameters

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

# numerical parameters
Nx = 1000  # number of points for hybridization integral
nec = 3  # limit for the hyb integral function
ddyson = 0.001
tmax = 20  # maximal time
N = 1000  # number of time points
times = linspace(0, tmax, N)


# define mathematical functions
def integral(G0, G, SE, times, t):
    total = 0
    d = times / N
    for x in times:
        if x < t:
            inside = 0
            for l in times:
                if l < x:
                    inside = inside + G0[t - x] * SE[x - l] * G[l]
            total = total + inside
    return total


def gamma(e):
    return 0.5 * ga / ((1 + exp(nu * (e - ec))) * (1 + exp(-nu * (e + ec))))


def f(e, mu):
    return 1 / (1 + exp(beta * (e - mu)))


def hyb_lesser(times, mu):
    H = zeros(N, complex)
    j = 0
    for t in times:
        x = linspace(-nec, nec, Nx)
        y = exp(1j * x * t) * gamma(x) * f(x, mu)
        H[j] = trapz(y, x)
        j = j + 1
    return H


hl = array([hyb_lesser(times, miu[0]), hyb_lesser(times, miu[1])])


def hyb_greater(t, mu):
    H = zeros(N, complex)
    j = 0
    for t in times:
        x = linspace(-nec, nec, Nx)
        y = exp(-1j * x * t) * gamma(x) * (1 - f(x, mu))
        H[j] = trapz(y, x)
        j = j + 1
    return H


hg = array([hyb_greater(times, miu[0]), hyb_greater(times, miu[1])])


def d(spin, role, i, j):  # 0=spin down 1=spin up, 0=annihilation 1=creation
    if spin == 0:
        if role == 0:
            if i == 0 and j == 1 or i == 1 and j == 3:
                return 1
            else:
                return 0
        if role == 1:
            if j == 0 and i == 1 or j == 1 and i == 3:
                return 1
            else:
                return 0
    if spin == 1:
        if role == 0:
            if i == 0 and j == 2 or i == 2 and j == 3:
                return 1
            else:
                return 0
        if role == 1:
            if j == 0 and i == 2 or j == 2 and i == 3:
                return 1
            else:
                return 0
    else:
        sys.exit("wrong parameters for d")


def cross_branch_hyb(a, b, t, lamb):
    S = 0
    for spin in [0, 1]:
        for lead in [0, 1]:
            S = S + (hyb_lesser(t, lead) * d(spin, 0, a, b) * d(spin, 1, b, a) * exp(-1j * lamb * t)
                     + hyb_greater(t, lead) * d(spin, 0, a, b) * d(spin, 1, b, a))


def g(t, i):  # t for time and j for the site number in the dot
    return exp(-1j * E[i] * t)


def update_green(times, SE, G, G0):
    S = zeros((4, N), complex)
    j = 0
    for t in times:
        for i in range(3):
            S[i, t] = G0[i, t] - integral(G0[i, :], G[i, :], SE[i, :], times, t)
        j = j + 1
    return S


def update_self_energy(times, G):
    S = zeros((4, N), complex)
    for a in range(3):
        j = 0
        for t in times:
            for spin in [0, 1]:
                for lead in [0, 1]:
                    for b in range(3):
                        S[a, j] = S[a, j] + (hl[lead, j] * d(spin, 0, a, b) * d(spin, 1, b, a)
                                             + hg[lead, j] * d(spin, 0, a, b) * d(spin, 1, b, a)) * G[b, j]
            j = j + 1
    return S


# build the initial states
G = zeros((4, N), complex)
G0 = zeros((4, N), complex)
SE = zeros((4, N), complex)
j = 0
for t in times:
    for i in range(3):
        o = g(t, i)
        G[i, j] = o
        G0[i, j] = o
    j = j + 1
SE = update_self_energy(times, G)
d = ddyson + 1

# start iterations to calculate G (green function), SE (self energy)
while d > ddyson:
    G_old = G
    G = update_green(times, SE, G_old, G0)
    SE = update_self_energy(times, G)
    d = LA.norm(G - G_old) / sqrt(4 * N) 
    print(d)
