import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

N_nodes = 50
max_radius = 150
h = max_radius / N_nodes

def hulthen_discrete(r, delta, orbital):
    global h
    t1 = -(delta * np.exp(-delta * r * h))/(1- np.exp(-delta * r * h))
    t2 = (orbital * (orbital + 1))/(2)
    t3 = delta / (1 - np.exp(-delta * r * h))
    t4 = np.exp(-delta * r * h)
    return t1 + (t2*(t3**2)*t4)

def hulthen_array(width = N_nodes, size = max_radius, orbital = 1, delta = 0.025):
    h = size / width
    A = np.diag(np.full(width, [1/(h**2)])) + np.diag(np.full(width-1, [-1/(2*(h**2))]), 1)\
    + np.diag(np.full(width-1, [-1/(2*(h**2))]), -1)
    for i in range(width):
        A[i, i] += -orbital*(orbital + 1)/((i+1)**2 * h**2) + hulthen_discrete(i+1, delta, orbital)
    A = np.delete(A, 0, 0)
    A = np.delete(A, -1, 0)
    A = np.delete(A, 0, 1)
    A = np.delete(A, -1, 1)
    return A

# plt.plot(np.arange(1, N_nodes), hulthen_discrete(np.arange(1, N_nodes), 0.025, 1))
# plt.show()

# arr = hulthen_array(width = 75, size = 40, orbital = 1, delta = 0.025)
# energy, wavefunction = eig(arr)
# energy.sort()
# print(energy)