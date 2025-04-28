import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

def hulthen_discrete(r: int, delta: float, currH: float, orb: int) -> float:
    '''
    :param r: index of which node in the discrete function is being considered
    :param delta: screening parameter
    :param currH: spacing between nodes of discrete function
    :return: approximation of Hulthen potential at the given radius with the given screening strength
    '''
    t1 = -delta * np.exp(- delta * (r * currH))/(1 - np.exp(- delta * (r * currH)))
    t2 = (orb * (orb + 1))/(2 * (r * currH)**2)
    return t1 + t2

def hulthen_array(width: int, size: float, orbital: int, delta: float) -> np.ndarray:
    '''
    :param width: number of nodes used to approximate the wavefunction; also the dimensions of the
    2D square array returned
    :param size: size of the system, in atomic units, out to which the wavefunction is approximated
    :param orbital: orbital quantum number
    :param delta: screening parameter
    :return: 2D square Hamiltonian matrix representing the system;
    Eigenevalues-eigenvector pairs represent energy-wavefunction pairs;
    Physically allowed states are those with E<0
    '''
    h = size / width
    offDiags = np.full(shape = width - 1, fill_value = -1/(2 * h**2)) # off-diagonal entries
    offDiagA = np.diag(offDiags, 1) + np.diag(offDiags, -1)

    diags = np.zeros(width)
    for i in range(len(diags)):
        true_index = i + 1
        diags[i] = (1/(h**2)) + (orbital * (orbital + 1))/(2 * (true_index * h)**2) + hulthen_discrete(true_index, delta, h, orb = orbital)
    diagA = np.diag(diags)
    A = offDiagA + diagA
    return A

def trapezoid_integral(wavefunction: np.ndarray, h: float) -> float:
    '''

    :param wavefunction: Any 1D array representing a discrete function
    :param h: spacing between nodes of discrete function
    :return: Approximation of integral over the given discrete function, using the composite trapezoid rule
    '''
    t1 = h
    t2 = (wavefunction[0] + wavefunction[-1])/2
    t3 = 0
    for i in range(1, len(wavefunction)):
        t3 += wavefunction[i]
    return t1 * (t2 + t3)

def normalize(wavefunction: np.ndarray, h: float) -> np.ndarray:
    '''
    :param wavefunction: Discrete function
    :param h: spacing between nodes of discrete function
    :return: factor which, when multiplied by each term in the discrete function, results in
    the square of that function having an enclosed area of 1. Useful for normalizing a discrete wavefunction approximation
    to create a valid probability density function
    '''

    return np.array(wavefunction) / np.sqrt(trapezoid_integral(wavefunction ** 2, h))

def first_derivative(wavefunction: np.ndarray, h: float) -> np.ndarray:
    '''

    :param wavefunction: vector representation of a discrete function as collection of points
    :param h: spacing between each node
    :return: discrete function which approximates the first derivative of 'wavefunction' parameter
    '''

    res = np.zeros_like(wavefunction)
    for i in range(1, len(wavefunction)-1):
        res[i] = (wavefunction[i+1] - wavefunction[i-1])/(2 * h)

    return res

def uncertainties(width: int, size: float, orb: int, delt: float, allowed_level: int) -> None:
    '''
    Function which adopts the architecture of the previous functions to calculate the uncertainty of the radius, and momentum,
    and the product of those two values. In atomic units, Heisenberg's uncertainty principle states that this product
    should always be greater than or equal to 0.5

    :param width: number of nodes used to approximate the wavefunction
    :param size: size of the system, in atomic units, out to which the wavefunction is approximated
    :param orb: orbital quantum number
    :param delt: Screening parameter
    :param allowed_level: Energy level of system to consider. Note that this is not necessarily the same as the principal
    quantum number; For example, there is no 1p orbital. Therefore, setting this value to 1 with an 'orb' value of 1
    will solve for the 2p orbital, since n = 2 is the lowest allowed energy level for the given orbital quantum number.
    :return: Prints the uncertainties for radius and momentum, and their product
    '''
    arr = hulthen_array(999, 50, orbital = orb, delta = delt)
    e, w = eig(arr)
    order = np.argsort(e)
    h_ex = size/width
    wave = normalize(wavefunction = w[:,order[allowed_level-1]], h = h_ex)
    r_range = np.linspace(wave[0], size, width)
    expectation_r = trapezoid_integral(wave**2 * r_range, h_ex)
    expectation_r_square = trapezoid_integral(wave**2 * r_range**2, h_ex)
    delta_r = np.sqrt(expectation_r_square - expectation_r**2)

    inner_derivative = first_derivative(wave/r_range, h_ex)
    expectation_p = trapezoid_integral(wave * r_range * inner_derivative, h_ex) # take the negative of this value's square
    expectation_p_square = - trapezoid_integral(wave/r_range * first_derivative(r_range**2 * inner_derivative, h_ex), h_ex)
    delta_p = np.sqrt(expectation_p_square + (expectation_p**2)) # flipped sign because expectation_p actually has a factor of i
    print('Delta r = ' + str(delta_r))
    print('Delta p = ' + str(delta_p))
    print('Uncertainty = ' + str(delta_r * delta_p))
    return

### Number of allowed energy levels as a function of screening parameter
# delta_range = np.linspace(0.025, 0.5, 40)
# num_levels = []
# for delt in delta_range:
#     curr_delt_levels = 0
#     for orb in range(1, 5):
#         arr = hulthen_array(999, 100, orbital = orb, delta = delt)
#         e, w = eig(arr)
#         allowed = [en for en in e if en < 0]
#         print(allowed)
#         curr_delt_levels += len(allowed)
#     num_levels.append(curr_delt_levels)
# plt.plot(delta_range, num_levels)
# plt.xlabel('Screening parameter $\delta$')
# plt.ylabel('Number of allowed energy levels found')
# plt.yticks(np.arange(0, 19), np.arange(0, 19))
# plt.axhline(0, color = 'grey', linestyle = '--')
# plt.axhline(1, color = 'grey', linestyle = '--')
# plt.axhline(2, color = 'grey', linestyle = '--')
# plt.axhline(3, color = 'grey', linestyle = '--')
# plt.axhline(4, color = 'grey', linestyle = '--')
# plt.axhline(6, color = 'grey', linestyle = '--')
# plt.axhline(10, color = 'grey', linestyle = '--')
# plt.axhline(18, color = 'grey', linestyle = '--')
# plt.show()

### What is the value of nodes at which values for energy converge? We will try varying node counts on
### a 2p shell in a box 100 wide, with delta = 0.025
# node_range = np.arange(10, 999, 30)
# num_levels = []
# for node in node_range:
#     curr_node = 0
#     for orb in range(1, 6):
#         arr = hulthen_array(node, 100, orbital = orb, delta = 0.1)
#         e, w = eig(arr)
#         allowed = [en for en in e if en < 0]
#         curr_node += len(allowed)
#     num_levels.append(curr_node)
# plt.plot(node_range, num_levels)
# plt.xlabel('Number of nodes used to approximate wavefunction')
# plt.ylabel('Number of allowed energy levels found')
# plt.show()

### Finding the lowest allowed energy level value for increasing numbers of nodes
# node_range = np.arange(10, 999, 1)
# lowest_levels = []
# for node in node_range:
#     arr = hulthen_array(node, 100, orbital = 1, delta = 0.1)
#     e, w = eig(arr)
#     lowest_levels.append(e[e.argsort(0)][0])
# plt.plot(node_range, lowest_levels)
# plt.xlabel('Number of nodes used to approximate wavefunction')
# plt.ylabel('Lowest allowed energy level found')
# plt.xscale('log')
# plt.show()


### Get ∆r∆p value for 2p, 3p, 4p, and 5p orbitals (delta = 0.025):
# uncertainties(999, 100, 1, 0.025, 1)
# uncertainties(999, 100, 1, 0.025, 2)
# uncertainties(999, 100, 1, 0.025, 3)
# uncertainties(999, 100, 1, 0.025, 4)

### Plot 3p orbital probability density approximations (normalized) for various screening parameters
### and their corresponding energies

# arr1 = hulthen_array(999, 50, 1, 0.025)
# e1, w1 = eig(arr1)
# order1 = np.argsort(e1)
# wave1 = normalize(wavefunction = w1[:,order1[0]], h = 100/999)
# plt.plot(wave1**2, label = '2p orbital')
# wave1_2 = normalize(wavefunction = w1[:,order1[1]], h = 100/999)
# plt.plot(wave1_2**2, label = '3p orbital')
# wave1_3 = normalize(wavefunction = w1[:,order1[2]], h = 100/999)
# plt.plot(wave1_3**2, label = '4p orbital')
# plt.title('Lowest three $\ell = 1$ energy level probability distributions')
# plt.xticks([0, 200, 400, 600, 800, 1000], [str(0), str(200*50/999)[:2], str(400*50/999)[:2], str(600*50/999)[:2], str(800*50/999)[:2], str(1000*50/999)[:2]])
# plt.xlabel('Radius (a.u.)')
# plt.ylabel('Probability')
# plt.legend()
# plt.show()
#
# arr2 = hulthen_array(999, 50, 1, 0.05)
# e2, w2 = eig(arr2)
# order2 = np.argsort(e2)
# wave2 = normalize(w2[:,order2[1]], 100/999)
#
# arr3 = hulthen_array(999, 50, 1, 0.075)
# e3, w3 = eig(arr3)
# order3 = np.argsort(e3)
# wave3 = normalize(w3[:,order3[1]], 50/999)
#
# arr4 = hulthen_array(999, 50, 1, 0.1)
# e4, w4 = eig(arr4)
# order4 = np.argsort(e4)
# wave4 = normalize(w4[:,order4[1]], 50/999)
#
# arr5 = hulthen_array(999, 50, 1, 0.125)
# e5, w5 = eig(arr5)
# order5 = np.argsort(e5)
# wave5 = normalize(w5[:,order5[1]], 50/999)
#
# arr6 = hulthen_array(999, 50, 1, 0.15)
# e6, w6 = eig(arr6)
# order6 = np.argsort(e6)
# wave6 = normalize(w6[:,order6[1]], 50/999)
#
# plt.plot(wave1**2, color = 'C0', label = '$\delta$ = 0.025', alpha = 0.9)
# plt.plot(wave2**2, color = 'C1',label = '$\delta$ = 0.050', alpha = 0.9, linestyle = '--')
# plt.plot(wave3**2, color = 'C2',label = '$\delta$ = 0.075', alpha = 0.9, linestyle = 'dotted')
# plt.plot(wave4**2, color = 'C3',label = '$\delta$ = 0.100', alpha = 0.9, linestyle = 'dashdot')
# plt.plot(wave5**2, color = 'C4',label = '$\delta$ = 0.125', alpha = 0.9, linestyle = (0, (1,1)))
# plt.plot(wave6**2, color = 'C5',label = '$\delta$ = 0.150', alpha = 0.9, linestyle = (0, (3,1,1,1)))
#
# plt.title('Normalized Probability Densities\nfor 3p Orbitals up to Distance 50', fontsize = 14)
# plt.xlabel('Node Index $r$')
# plt.ylabel('Probability')
# plt.legend(fontsize = 11)
# plt.show()
#
# plt.axhline(e6[order6[1]], color = 'C5', label = '$\delta$ = 0.150, E = %.5f' % e6[order6][1], linestyle = (0, (3,1,1,1)))
# plt.axhline(e5[order5[1]], color = 'C4', label = '$\delta$ = 0.125, E = %.5f' % e5[order5][1], linestyle = (0, (1,1)))
# plt.axhline(e4[order4[1]], color = 'C3', label = '$\delta$ = 0.100, E = %.5f' % e4[order4][1], linestyle = 'dashdot')
# plt.axhline(e3[order3[1]], color = 'C2', label = '$\delta$ = 0.075, E = %.5f' % e3[order3][1], linestyle = 'dotted')
# plt.axhline(e2[order2[1]], color = 'C1', label = '$\delta$ = 0.050, E = %.5f' % e2[order2[1]], linestyle = '--')
# plt.axhline(e1[order1[1]], color = 'C0', label = '$\delta$ = 0.025, E = %.5f' % e1[order1[1]])
# plt.axhline(0, color = 'k')
# plt.legend(fontsize = 11)
# plt.xticks([])
# plt.ylim(-.046, 0.001)
# plt.title('Relative Energies of Approximated Wavefunctions\nfor Various Screening Parameters', fontsize = 14)
# plt.show()