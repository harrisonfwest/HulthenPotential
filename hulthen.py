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

def trapezoid_integral(wavefunction, h):
    t1 = h
    t2 = (wavefunction[0] + wavefunction[-1])/2
    t3 = 0
    for i in range(1, len(wavefunction)):
        t3 += wavefunction[i]
    return t1 * (t2 + t3)

def normalize(wavefunction, h): # returns a (non-squared) normalized version of the (non-squared) input wavefunction
    wave2 = wavefunction.copy()
    norm = 1 / np.sqrt(trapezoid_integral(wavefunction ** 2, 100 / 999))
    for i in range(len(wave2)):
        wave2[i] *= norm
    return wave2

arr1 = hulthen_array(999, 50, 1, 0.025)
e1, w1 = eig(arr1)
order1 = np.argsort(e1)
wave1 = normalize(wavefunction = w1[:,order1[1]], h = 100/999)

arr2 = hulthen_array(999, 50, 1, 0.05)
e2, w2 = eig(arr2)
order2 = np.argsort(e2)
wave2 = normalize(w2[:,order2[1]], 100/999)

arr3 = hulthen_array(999, 50, 1, 0.075)
e3, w3 = eig(arr3)
order3 = np.argsort(e3)
wave3 = normalize(w3[:,order3[1]], 50/999)

arr4 = hulthen_array(999, 50, 1, 0.1)
e4, w4 = eig(arr4)
order4 = np.argsort(e4)
wave4 = normalize(w4[:,order4[1]], 50/999)

arr5 = hulthen_array(999, 50, 1, 0.125)
e5, w5 = eig(arr5)
order5 = np.argsort(e5)
wave5 = normalize(w5[:,order5[1]], 50/999)

arr6 = hulthen_array(999, 50, 1, 0.15)
e6, w6 = eig(arr6)
order6 = np.argsort(e6)
wave6 = normalize(w6[:,order6[1]], 50/999)

plt.plot(wave1**2, color = 'C0', label = '$\delta$ = 0.025', alpha = 0.9)
plt.plot(wave2**2, color = 'C1',label = '$\delta$ = 0.050', alpha = 0.9, linestyle = '--')
plt.plot(wave3**2, color = 'C2',label = '$\delta$ = 0.075', alpha = 0.9, linestyle = 'dotted')
plt.plot(wave4**2, color = 'C3',label = '$\delta$ = 0.100', alpha = 0.9, linestyle = 'dashdot')
plt.plot(wave5**2, color = 'C4',label = '$\delta$ = 0.125', alpha = 0.9, linestyle = (0, (1,1)))
plt.plot(wave6**2, color = 'C5',label = '$\delta$ = 0.150', alpha = 0.9, linestyle = (0, (3,1,1,1)))

plt.title('Normalized Probability Densities for 3p Orbitals up to Distance 50')
plt.xlabel('Node Index $r$')
plt.ylabel('Probability')
plt.legend()
plt.show()

plt.axhline(e6[order6[1]], color = 'C5', label = '$\delta$ = 0.150, E = %.5f' % e6[order6][1], linestyle = (0, (3,1,1,1)))
plt.axhline(e5[order5[1]], color = 'C4', label = '$\delta$ = 0.125, E = %.5f' % e5[order5][1], linestyle = (0, (1,1)))
plt.axhline(e4[order4[1]], color = 'C3', label = '$\delta$ = 0.100, E = %.5f' % e4[order4][1], linestyle = 'dashdot')
plt.axhline(e3[order3[1]], color = 'C2', label = '$\delta$ = 0.075, E = %.5f' % e3[order3][1], linestyle = 'dotted')
plt.axhline(e2[order2[1]], color = 'C1', label = '$\delta$ = 0.050, E = %.5f' % e2[order2[1]], linestyle = '--')
plt.axhline(e1[order1[1]], color = 'C0', label = '$\delta$ = 0.025, E = %.5f' % e1[order1[1]])
plt.axhline(0, color = 'k')
plt.legend()
plt.xticks([])
plt.ylim(-.046, 0.001)
plt.title('Relative Energies of Approximated Wavefunctions\nfor Various Screening Parameters')
plt.show()