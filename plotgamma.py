import numpy as np
import matplotlib.pyplot as plt



def gamma(tb, tm, eps_l, w):
    temp = 0
    if abs(w - eps_l) < (2 * tb):
        # temp=ga
        temp = ((tm ** 2) / (2 * tb ** 2)) * np.sqrt(4 * tb ** 2 - (w - eps_l) ** 2)
    return temp

def sketch_gamma(tl, tm, miu):
    for e in (miu):
        for i in range(1, 6):
            tb = i * tl
            ga = (tm ** 2) / tb
            w = np.linspace(-200*ga, 200*ga, 1000)
            gam = np.zeros(1000)
            for j in range(1000):
                gam[j] = gamma(tb, tm, e, w[j])
            gam = gam / ga
            w = w / ga
            plt.plot(w, gam, label="tb={}".format(tb))
            plt.legend()
        plt.xlabel("\u03C9/\u0393")
        plt.ylabel("\u0393(\u03C9)/\u0393")
    plt.show()
ga=1
tl = 10
miu = [-20, 20]
tm = np.sqrt(tl * ga)
miu = miu*ga

sketch_gamma(tl, tm, miu)
