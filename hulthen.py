import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

h = 0.5
max_radius = 100



def hulthen_discrete(r, delta):
    return delta * np.exp(-delta * r * h)/(1 - np.exp(-delta * r * h))

def hulthen_array(width = int(max_radius/h), orbital = 1, delta = 0.025):
    A = np.diag(np.full(width, [1/(h**2)])) + np.diag(np.full(width-1, [-1/(2*(h**2))]), 1)\
    + np.diag(np.full(width-1, [-1/(2*(h**2))]), -1)
    for i in range(width):
        A[i, i] += -orbital*(orbital + 1)/((i+1) * h)**2 + hulthen_discrete(i+1, delta)
    return A


arr = hulthen_array()
energy, wavefunction = eig(arr)

# Plot the first excited state radial wave-function
data = (wavefunction[1]**2)[1::]
for i in range(len(data)):
    data[i] /= (i+1)
plt.plot(data)
plt.title('Squared wavefunction for a 2p electron')
plt.show()