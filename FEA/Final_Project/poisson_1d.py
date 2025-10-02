import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
pi=np.pi

def exact(x):
    return np.sin(pi*x)

L = 1 # Assumed in derivation of manufactured solution
u_l, u_r = 0, 0
N = 5
h = L/N


def create_elemental_K():
    K_e=np.array([[1,-1],[-1,1]])
    return K_e

def create_elemental_f(x_a):
    f = np.zeros(2)
    forcing_function = lambda xbar: pi**2 * np.sin(pi*(xbar+x_a)) # we want the function to go from 0 to h
    f_phi_1 = lambda x: (1-x/h)*forcing_function(x)
    f_phi_2 = lambda x: (x/h)*forcing_function(x)
    f[0] = quad(f_phi_1, 0, h)[0]
    f[1] = quad(f_phi_2, 0, h)[0]
    return f
