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
    return t1 #+ (t2*(t3**2)*t4) # omitting extra terms, using pure Hulthen potential

# ## Plot of Hulthen Potential: ###
# plt.plot(np.linspace(1, 200, 10000), \
#          hulthen_discrete(np.linspace(1, 200, 10000), 0.025, 1, currH = 1))
# plt.xscale('log')
# plt.show()

def hulthen_array(width = N_nodes, size = max_radius, orbital = 1, delta = 0.025):
    h = size / width
    offDiags = np.full(shape = width - 1, fill_value = -1/(2 * h**2)) # off-diagonal entries
    offDiagA = np.diag(offDiags, 1) + np.diag(offDiags, -1)

    diags = np.zeros(width)
    for i in range(len(diags)):
        true_index = i + 1
        diags[i] = (1/(h**2)) + (orbital * (orbital + 1))/(2 * (true_index * h)**2) + hulthen_discrete(true_index, delta, orbital, h)
    diagA = np.diag(diags)
    A = offDiagA + diagA
    return A

arr = hulthen_array(width = 999, size = 100, orbital = 1, delta = 0.025)
e, w = eig(arr)
ordered_e = np.argsort(e)

# for i in ordered_e[:6]:
#     plt.plot(w[:,i]**2)
#     plt.title('E = ' + str(e[i]) + ' (l = 1)' + ', spot ' + str(i))
#     plt.show()

def simpson_integral(wavefunction, h):
    length = int(len(wavefunction))
    half_length = int(length / 2)

    t1 = (1/3) * h

    t2 = wavefunction[0]**2 + wavefunction[-1]**2

    t3 = 0
    for i in range(1, half_length):
        t3 += wavefunction[2*i - 1]**2 + 2
    t3 *= 4

    t4 = 0
    for i in range(1, half_length - 1):
        t4 += wavefunction[2 * i]**2
    t4 *= 2

    return t1 * (t2 + t3 + t4)

func = w[:,866]
plt.plot(func)
plt.show()
norm = 1/np.sqrt(simpson_integral(func, 100/999))
# print(norm)
for i in range(len(func)):
    func[i] *= norm
plt.plot(func**2)
plt.show()