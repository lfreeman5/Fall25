import sympy as sp
import numpy as np

x = sp.Symbol('x')

h1 = 10
h2 = 5
EI1 = 2*2e6
EA2 = 2e6
phis = [
        1-3*(x/h1)**2 + 2*(x/h1)**3,
        -x*(1-x/h1)**2,
        3*(x/h1)**2 -2*(x/h1)**3,
        -x*((x/h1)**2-x/h1)
    ]

dphis = [sp.diff(p,x,2) for p in phis]

k_e1 = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        k_e1[i,j] = sp.integrate(dphis[i]*dphis[j],(x,0,h1))
k_e1 *= EI1
print(k_e1 /2 * h1**3 / EI1)
print(k_e1/1e3)

f_e1 = np.zeros(4)
for i in range(4):
    f_e1[i] = sp.integrate(-100*phis[i],(x,0,h1))
print(f_e1)

k_e2 = np.array([[1.,-1.],[-1.,1.]])
k_e2 *= EA2/h2/1.
print(k_e2/1e3)

k_g = np.zeros((5,5))
k_g[0:4,0:4] += k_e1
k_g[0,0]+= k_e2[0,0]
k_g[-1,0] += k_e2[1,0]
k_g[0,-1] += k_e2[0,1]
k_g[-1,-1] += k_e2[1,1]

f_g = np.zeros(5)
f_g[0:4] = f_e1

k_g_mod = np.eye(5)

k_g_mod[0:2,0:2] = k_g[0:2,0:2]
f_g_mod = np.zeros(5)
f_g_mod[0:2] = f_g[0:2]

u_g = np.linalg.solve(k_g_mod, f_g_mod)
print(u_g)
Q=k_g@u_g - f_g
print(Q)