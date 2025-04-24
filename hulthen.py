import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

N_nodes = 50
max_radius = 150

def hulthen_discrete(r, delta, orbital, currH):
    t1 = -(delta * np.exp(-delta * r * currH))/(1- np.exp(-delta * r * currH))
    t2 = (orbital * (orbital + 1))/(2)
    t3 = delta / (1 - np.exp(-delta * r * currH))
    t4 = np.exp(-delta * r * currH)
    return t1 + (t2*(t3**2)*t4)

def hulthen_array(width = N_nodes, size = max_radius, orbital = 1, delta = 0.025):
    true_size = width - 2 # we omit the first/last rows and left/rightmost columns,
                          # since we define u(r) = 0 at the first and last nodes
    h = size / width
    A = np.diag(np.full(true_size, [1/(h**2)]))                 \
      + np.diag(np.full(true_size-1, [-1/(2*(h**2))]), 1)    \
      + np.diag(np.full(true_size-1, [-1/(2*(h**2))]), -1)
    for i in range(true_size):
        true_r = i + 2
        A[i, i] += (orbital*(orbital + 1))/(true_r * h)**2 + hulthen_discrete(true_r, delta, orbital, h)
    return A

### Plot of Hulthen Potential: ###
# plt.plot(np.linspace(1, 200, 10000), \
#          hulthen_discrete(np.linspace(1, 200, 10000), 0.025, 1, currH = 1))
# plt.xscale('log')
# plt.show()


# Finding u(r) for 3p electron with screening parameter 0.025
arr = hulthen_array(width = 60, size = 200, orbital = 1, delta = 0.025)
e, w = eig(arr)

# for wave in w:
#     plt.plot(wave**2, alpha = 1)
#     plt.show()
#     wave[0] = 0

sorted_e, sorted_w = zip(*sorted(zip(e, w)))
# plt.plot(sorted_w[1]**2)
func = np.append(0, np.append(sorted_w[3], 0)) # define first and last nodes as 0
plt.plot(np.arange(0, 200, 200/60), func**2) # arange args are 0, size, size/width
plt.show()