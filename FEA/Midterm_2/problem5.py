import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

x = sp.Symbol('x')

h = 5
EI = 1
p = lambda x: 500/h*x

phis = [
        1-3*(x/h)**2 + 2*(x/h)**3,
        -x*(1-x/h)**2,
        3*(x/h)**2 -2*(x/h)**3,
        -x*((x/h)**2-x/h)
    ]
dphis = [sp.diff(p,x,2) for p in phis]

k_e = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        k_e[i,j] = sp.integrate(dphis[i]*dphis[j],(x,0,h))
k_e *= EI
print(k_e /2 * h**3 / EI)
print(k_e*1e2)

f_e1 = np.zeros(4)
for i in range(4):
    f_e1[i] = sp.integrate(p(x)*phis[i],(x,0,h))
print(f_e1)

k_g = np.zeros((8,8))
k_g[0:4,0:4] += k_e
k_g[2:6,2:6] += k_e
k_g[4:8,4:8] += k_e
f_g = np.zeros(8)
f_g[0:4] += f_e1
print('K_G:')
for r in 1e2*k_g:
    print(' '.join(f"{val:12.4f}" for val in r))

k_g_mod = np.copy(k_g)
k_g_mod[0,:] = 0.0
k_g_mod[0,0] = 1.
k_g_mod[6,:] = 0.0
k_g_mod[6,6] = 1.
rhs_vector = np.copy(f_g)
rhs_vector[4] = 1000.
rhs_vector[0]=0
rhs_vector[6] = 0

u_g = np.linalg.solve(k_g_mod, rhs_vector)
print(u_g)

Q_g = k_g@u_g - f_g
print(Q_g)

# Plot u(x) using the shape functions and nodal values
x_vals = np.linspace(0, 15, 300)
u_vals = np.zeros_like(x_vals)

# There are 3 elements, each of length h=5
for e in range(3):
    # Element node indices
    n0 = 2*e
    n1 = 2*e+1
    n2 = 2*e+2
    n3 = 2*e+3
    print(f'indices for element {e+1}: {n0+1} {n1+1} {n2+1} {n3+1}')
    # Local nodal values
    u_e = u_g[n0:n3+1]
    # Local x in [0, h]
    mask = (x_vals >= e*h) & (x_vals <= (e+1)*h)
    x_local = x_vals[mask] - e*h
    # Evaluate shape functions at x_local
    phi_e = [
        sp.lambdify(x, phis[0], 'numpy')(x_local),
        sp.lambdify(x, phis[1], 'numpy')(x_local),
        sp.lambdify(x, phis[2], 'numpy')(x_local),
        sp.lambdify(x, phis[3], 'numpy')(x_local)
    ]
    u_vals[mask] = u_e[0]*phi_e[0] + u_e[1]*phi_e[1] + u_e[2]*phi_e[2] + u_e[3]*phi_e[3]

plt.figure()
plt.plot(x_vals, u_vals, label='FEA $v(x)$')
plt.xlabel('x')
plt.ylabel('Transverse Displacement $v(x)$')
plt.title('FEA Solution for $v(x)$')
plt.grid(True)
plt.legend()
plt.show()