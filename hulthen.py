import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

N_nodes = 15
max_radius = 40
h = int(max_radius / N_nodes)


def hulthen_discrete(r, delta):
    return delta * np.exp(-delta * r * h)/(1 - np.exp(-delta * r * h))

def hulthen_array(width = N_nodes, orbital = 1, delta = 0.025):
    A = np.diag(np.full(width, [1/(h**2)])) + np.diag(np.full(width-1, [-1/(2*(h**2))]), 1)\
    + np.diag(np.full(width-1, [-1/(2*(h**2))]), -1)
    for i in range(width):
        A[i, i] += -orbital*(orbital + 1)/((i+1)**2 * h**2) + hulthen_discrete(i+1, delta)
    return A


arr = hulthen_array(width = 25, orbital = 1, delta = 0.025)
energy, wavefunction = eig(arr)

data = (wavefunction[3])
for i in range(len(data)):
    data[i] /= ((i+1) * h)
data = (data**2)[1::]
plt.plot(data)
plt.show()