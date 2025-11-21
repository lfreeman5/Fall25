import numpy as np
import sympy as sp

def print_matrix(mat, name):
    print(f"{name}:")
    for row in mat:
        print(" ".join(f"{val:10.4f}" for val in row))
    print()


a,b=4,4
x = sp.Symbol('x')
y = sp.Symbol('y')

phi1 = (1 - x/a) * (1 - y/b)
phi2 = (x/a) * (1 - y/b)
phi3 = (x/a) * (y/b)
phi4 = (1 - x/a) * (y/b)

phis = [phi1, phi2, phi3, phi4]

S11 = np.zeros((4,4))
S12 = np.zeros((4,4))
S21 = np.zeros((4,4))
S22 = np.zeros((4,4))
S00 = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        dphi_i_dx = sp.diff(phis[i], x)
        dphi_j_dx = sp.diff(phis[j], x)
        dphi_i_dy = sp.diff(phis[i], y)
        dphi_j_dy = sp.diff(phis[j], y)
        s11_integrand = dphi_i_dx * dphi_j_dx
        s12_integrand = dphi_i_dx * dphi_j_dy
        s21_integrand = dphi_i_dy * dphi_j_dx
        s22_integrand = dphi_i_dy * dphi_j_dy
        s00_integrand = phis[i] * phis[j]
        S11[i, j] = float(sp.integrate(s11_integrand, (x, 0, a), (y, 0, b)))
        S12[i, j] = float(sp.integrate(s12_integrand, (x, 0, a), (y, 0, b)))
        S21[i, j] = float(sp.integrate(s21_integrand, (x, 0, a), (y, 0, b)))
        S22[i, j] = float(sp.integrate(s22_integrand, (x, 0, a), (y, 0, b)))
        S00[i, j] = float(sp.integrate(s00_integrand, (x, 0, a), (y, 0, b)))

print_matrix(36*S00/(a*b), 'S00')
print_matrix(6*S11, 'S11')
print_matrix(4*S12, 'S12')
print_matrix(4*S21, 'S21')
print_matrix(6*S22, 'S22')

print_matrix(S00+S11+S12+S21+S22, 'S_All')
