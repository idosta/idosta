from matplotlib import colors
from numpy import *
import matplotlib.pyplot as plt

a = real(load('p_nca.npy'))
b = real(load('p_qme.npy'))


for i in [0, 1, 3]:
    plt.imshow(a[i].T, cmap='Reds')
    plt.colorbar()
    plt.show()
    plt.imshow(b[i].T, cmap='Reds')
    print(amax(abs((a[i] - b[i]))))
    plt.colorbar()
    plt.show()
    plt.imshow((a[i].T - b[i].T), cmap='Reds')
    plt.colorbar()
    plt.show()

