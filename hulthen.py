import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

h = 1
delta = 0.025

def hulthen_discrete(r):
    return delta * np.exp(-delta * r * h)/(1 - np.exp(-delta * r * h))

def hulthen_array(width = 5, orbital = 2):
    A = np.diag(np.full(width, [1/(h**2)])) + np.diag(np.full(width-1, [-1/(2*(h**2))]), 1)\
    + np.diag(np.full(width-1, [-1/(2*(h**2))]), -1)
    for i in range(width):
        A[i, i] += -orbital*(orbital + 1)/((i+1) * h)**2 + hulthen_discrete(i+1)
    return A

arr = hulthen_array(50)
print(arr)

energy, wavefunction = eig(arr)
level = 1
print(energy[level-1])
print(wavefunction[level-1])
plt.plot(wavefunction[0]**2, label = 'Ground state')
plt.plot(wavefunction[1]**2, label = 'Excited state 1')
plt.plot(wavefunction[2]**2, label = 'Excited state 2')
plt.plot(wavefunction[3]**2, label = 'Excited state 3')
plt.legend()
plt.show()