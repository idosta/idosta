from matplotlib import colors
from numpy import *
import matplotlib.pyplot as plt


def S(M):
    p = where(M != 0)
    M = M[min(p[0]): max(p[0]) + 1, min(p[1]): max(p[1]) + 1]
    for x in argwhere(M == 0.):
        M[x[0], x[1]] = 0.25 * (
                M[x[0] + 1, x[1] + 1] + M[x[0] - 1, x[1] - 1] + M[x[0] - 1, x[1] + 1] + M[x[0] + 1, x[1] - 1])
    return M


lamb = 0.0001  # remember to update
t_l = 1
t_m = 1
u = 8
dim = 1
T = 0.1
T2 = 2
dv = 0.02
V = around(arange(0, 6.2, 0.2), 1)
H = arange(0, 5.25, 0.25)

I = load('/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l)
         + '_t_m' + str(t_m) + '_u' + str(u) + '/I_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose()
N = load('/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l)
         + '_t_m' + str(t_m) + '_u' + str(u) + '/noise_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose()
NT = load('/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l)
          + '_t_m' + str(t_m) + '_u' + str(u) + '/noise2_map_tl_' + str(t_l) + 'tm_' + str(
    t_m) + '.out.npy').transpose()


#
# N2 = load('/home/ido/gcohenlab/nca/nca_T' + str(T2) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l)
# + '_t_m' + str(t_m) + '_u' + str(u) + '/noise_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose() I2 =
# load('/home/ido/gcohenlab/nca/nca_T' + str(T2) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l) +
# '_t_m' + str(t_m) + '_u' + str(u) + '/I_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose() I3 = load(
# '/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(3) + '_t_l' + str(t_l) + '_t_m' +
# str(t_m) + '_u' + str(u) + '/I_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose() N3 = load(
# '/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(3) + '_t_l' + str(t_l) + '_t_m' +
# str(t_m) + '_u' + str(u) + '/noise_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose() I = S(load(
# '/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l) + '_t_m' +
# str(t_m) + '_u' + str(u) + '/I_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose()) N = S(load(
# '/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l) + '_t_m' +
# str(t_m) + '_u' + str(u) + '/noise_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose()) NT = S(load(
# '/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l) + '_t_m' +
# str(t_m) + '_u' + str(u) + '/noise2_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy') #        .transpose()) NT2
# = load('/home/ido/gcohenlab/nca/nca_T' + str(T2) + '_d_lamb' + str(lamb) + '_dim' + str(dim) + '_t_l' + str(t_l) +
# '_t_m' + str(t_m) + '_u' + str(u) + '/noise2_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose() NT3 =
# load('/home/ido/gcohenlab/nca/nca_T' + str(T) + '_d_lamb' + str(lamb) + '_dim' + str(3) + '_t_l' + str(t_l) +
# '_t_m' + str(t_m) + '_u' + str(u) + '/noise2_map_tl_' + str(t_l) + 'tm_' + str(t_m) + '.out.npy').transpose()
def der_cal(a):
    da = zeros(shape(a))
    for i in range(1, len(a[0, :]) - 1):
        da[:, i] = (a[:, i + 1] - a[:, i - 1]) / (dv * 2)
    da[:, 0] = da[:, 1]
    da[:, len(I[0, :]) - 1] = da[:, len(I[0, :]) - 2]
    dda = zeros(shape(a))
    for i in range(1, len(a[0, :]) - 1):
        dda[:, i] = (a[:, i + 1] + a[:, i - 1] - 2 * a[:, i]) / (dv ** 2)
    dda[:, 0] = dda[:, 1]
    dda[:, len(I[0, :]) - 1] = dda[:, len(I[0, :]) - 2]
    return da, dda


dI, ddI = der_cal(I)

plt.matshow(I, extent=[0, 6.20, 0, 5.25], cmap='PiYG', interpolation='bilinear')
plt.colorbar()
plt.xlabel("v")
plt.ylabel("gate")
plt.show()
plt.matshow(dI, extent=[0, 6.2, 0, 5.25], cmap='PiYG')
plt.colorbar()
plt.xlabel("v")
plt.ylabel("gate")
plt.show()
plt.matshow(ddI, extent=[0, 6.2, 0, 5.25], cmap='PiYG')
plt.colorbar()
plt.xlabel("v")
plt.ylabel("gate")
plt.show()
