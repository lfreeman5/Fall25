import numpy as np
import sympy as sp

h=1./20
x = sp.Symbol('x')
phis = [1-x/h,x/h]
dphis = [sp.diff(p,x,1) for p in phis]


ke = np.zeros((2,2))
for i in range(2):
    for j in range(2):
        ke[i,j] += sp.integrate(dphis[i]*dphis[j],(x,0,h))
        ke[i,j] += sp.integrate(144*phis[i]*phis[j],(x,0,h))

k_g = np.zeros((3,3))
k_g[0:2,0:2] += ke
k_g[1:3,1:3] += ke
print(ke)
print(k_g)

k_g[2,2]+=3
k_g[0,:] = np.array([1,0,0])
print(k_g)
Q = np.zeros(3)
Q[0]=400

u=np.linalg.solve(k_g,Q)
print(u)