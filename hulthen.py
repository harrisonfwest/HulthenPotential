import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

N_nodes = 50
max_radius = 60
h = max_radius / N_nodes

def hulthen_discrete(r, delta):
    return delta * np.exp(-delta * r * h)/(1 - np.exp(-delta * r * h))

def hulthen_array(width = N_nodes, size = max_radius, orbital = 1, delta = 0.025):
    h = size / width
    A = np.diag(np.full(width, [1/(h**2)])) + np.diag(np.full(width-1, [-1/(2*(h**2))]), 1)\
    + np.diag(np.full(width-1, [-1/(2*(h**2))]), -1)
    for i in range(width):
        A[i, i] += -orbital*(orbital + 1)/((i+1)**2 * h**2) + hulthen_discrete(i+1, delta)
    A[0, 0] = 0
    A[1, 0] = 0
    A[-2, -1]
    A[-1, -1] = 0
    print(A)
    return A


arr = hulthen_array(width = 20, size = 40, orbital = 1, delta = 0.025)
energy, wavefunction = eig(arr)
data = (wavefunction[2]**2)
plt.plot(data)
plt.show()