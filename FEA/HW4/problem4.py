import sympy as sp
import numpy as np

x=sp.Symbol('x')
h=1/2
phiq = [(1-2*x/h)*(1-x/h), 4*x/h*(1-x/h), -x/h*(1-2*x/h)]
dphiq = [sp.diff(phi,x) for phi in phiq]

K = sp.zeros(3)
for i in range(3):
    for j in range(3):
        K[i,j] = sp.integrate(dphiq[i]*dphiq[j], (x, 0, 1/2))

print(K)
K = np.array(K)
F = np.array([sp.integrate(phi,(x,0,1/2)) for phi in phiq])
print(F*3)

# Local element stiffness matrix and force vector
K_local = np.array(K, dtype=float)
F_local = np.array([sp.integrate(phi,(x,0,1/2)) for phi in phiq], dtype=float)

# Assemble global K_G (5x5) and f_g (size 5) for 2 elements
K_G = np.zeros((5,5))
f_g = np.zeros(5)

# Element 1: nodes 0-1-2
K_G[0:3,0:3] += K_local
f_g[0:3] += F_local

# Element 2: nodes 2-3-4
K_G[2:5,2:5] += K_local
f_g[2:5] += F_local

print("Global stiffness matrix K_G:")
print(K_G)
print("Global force vector f_g:")
print(f_g)

K_G[0,:]=0.
K_G[0,0]=1
f_g[0]=0.

u=np.linalg.solve(K_G,f_g)
print(u)

# Piecewise quadratic interpolation polynomial for u
# Element 1: x in [0, h], nodes 0,1,2
# Element 2: x in [h, 2*h], nodes 2,3,4

u0, u1, u2, u3, u4 = sp.symbols('u0 u1 u2 u3 u4')
phiq = [(1-2*x/h)*(1-x/h), 4*x/h*(1-x/h), -x/h*(1-2*x/h)]

# Interpolation for element 1 (x in [0, h])
u_elem1 = u[0]*phiq[0] + u[1]*phiq[1] + u[2]*phiq[2]

# Interpolation for element 2 (x in [h, 2*h])
x2 = x - h  # local coordinate for element 2
phiq2 = [(1-2*x2/h)*(1-x2/h), 4*x2/h*(1-x2/h), -x2/h*(1-2*x2/h)]
u_elem2 = u[2]*phiq2[0] + u[3]*phiq2[1] + u[4]*phiq2[2]

u_interp = sp.Piecewise(
    (u_elem1, (x >= 0) & (x <= h)),
    (u_elem2, (x > h) & (x <= 2*h))
)
u_interp_prime = sp.diff(u_interp, x)

print("u_interp (symbolic):")
print(u_interp)
print("u_interp_prime (symbolic):")
sp.pprint(u_interp_prime)
