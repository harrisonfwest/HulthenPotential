import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt
import pandas as pd

N_nodes = 50
max_radius = 150
h = max_radius / N_nodes

def hulthen_discrete(r, delta, orbital, currH):
    t1 = -(delta * np.exp(-delta * r * currH))/(1- np.exp(-delta * r * currH))
    t2 = (orbital * (orbital + 1))/(2)
    t3 = delta / (1 - np.exp(-delta * r * currH))
    t4 = np.exp(-delta * r * currH)
    return t1 + (t2*(t3**2)*t4)

def hulthen_array(width = N_nodes, size = 1, orbital = 1, delta = 0.025):
    h = size / width
    A = np.diag(np.full(width-1, [1/(h**2)])) + np.diag(np.full(width-2, [-1/(2*(h**2))]), 1)\
    + np.diag(np.full(width-2, [-1/(2*(h**2))]), -1)
    for i in range(width-1):
        A[i, i] += -orbital*(orbital + 1)/((i+2)**2 * h**2) + hulthen_discrete(i+2, delta, orbital, h)
    return A

plt.plot(np.linspace(.25, 200, 10000), \
         hulthen_discrete(np.linspace(.25, 200, 10000), 0.025, 1, currH = (200-.25)/10000))
plt.xscale('log')
plt.show()

arr = hulthen_array(width = 40, size = 40, orbital = 1, delta = 0.025)
e, w = eig(arr)
for wave in w:
    wave[0] = 0
sorted_e, sorted_w = zip(*sorted(zip(e, w)))
func = np.append(0, np.append(sorted_w[1], 0))
plt.plot(func**2)
plt.show()