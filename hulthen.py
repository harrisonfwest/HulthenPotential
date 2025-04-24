import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

N_nodes = 50
max_radius = 150

def hulthen_discrete(r, delta, orbital, currH):
    t1 = -(delta * np.exp(-delta * r * currH))/(1- np.exp(-delta * r * currH))
    t2 = (orbital * (orbital + 1))/2
    t3 = delta / (1 - np.exp(-delta * r * currH))
    t4 = np.exp(-delta * r * currH)
    return t1 #+ (t2*(t3**2)*t4)

def hulthen_array(width = N_nodes, size = max_radius, orbital = 1, delta = 0.025):
    true_size = width - 2
    h = size / width
    offDiags = np.full(shape = true_size - 1, fill_value = -1/(2 * h**2)) # off-diagonal entries
    offDiagA = np.diag(offDiags, 1) + np.diag(offDiags, -1)

    diags = np.zeros(true_size)
    for i in range(len(diags)):
        true_index = i + 2
        diags[i] = (1/(h**2)) + (orbital * (orbital + 1))/(2 * (true_index * h)**2) + hulthen_discrete(true_index, delta, orbital, h)
    diagA = np.diag(diags)
    A = offDiagA + diagA
    return A

### Plot of Hulthen Potential: ###
# plt.plot(np.linspace(1, 200, 10000), \
#          hulthen_discrete(np.linspace(1, 200, 10000), 0.025, 1, currH = 1))
# plt.xscale('log')
# plt.show()

# Eigenenergies for l = 1 (p shell) electrons
arr = hulthen_array(width = 2000, size = 100, orbital = 1, delta = 0.025)
e, w = eig(arr)
sorted_e, sorted_w = zip(*sorted(zip(e, w)))
print('Eigenenergies of p orbital for n = 2, 3, 4...:' + str(sorted_e))
# returns same energies as Table 1 (Varshni 1990) with original pure Hulthen potential
# but not with the additional terms from Eq. 5 (same paper)