import numpy as np
import sympy as sp

x = sp.Symbol('x')
h1 = 1
phis = [
        1-3*(x/h1)**2 + 2*(x/h1)**3,
        -x*(1-x/h1)**2,
        3*(x/h1)**2 -2*(x/h1)**3,
        -x*((x/h1)**2-x/h1)
    ]
dphis = [sp.diff(p,x,2) for p in phis]
dphi_funcs = [sp.lambdify(x, dphi, modules='numpy') for dphi in dphis]

nc2_pts = [0,1]
nc2_wts = [0.5,0.5]
nc4_pts = [0,1./3,2./3,1.]
nc4_wts = [1./8, 3./8, 3./8, 1./8]

ke_nc2 = np.zeros((4,4))
ke_nc4 = np.zeros((4,4))
for i in range(4):
    for j in range(4):
        f = lambda x: dphi_funcs[i](x) * dphi_funcs[j](x)
        for k,p in enumerate(nc2_pts):
            ke_nc2[i,j] += f(p)*nc2_wts[k]
        for k,p in enumerate(nc4_pts):
            ke_nc4[i,j] += f(p)*nc4_wts[k]

print(ke_nc2)
print(ke_nc4)
ke_gl2 = np.zeros((4,4))
ke_gl4 = np.zeros((4,4))
gl2_pts = [-1/3**0.5, 1/3**0.5]
gl2_wts = [1.0, 1.0]

gl4_pts = [
    -0.8611363115940526,
    -0.3399810435848563,
     0.3399810435848563,
     0.8611363115940526
]

gl4_wts = [
    0.3478548451374538,
    0.6521451548625461,
    0.6521451548625461,
    0.3478548451374538
]

for i in range(4):
    for j in range(4):
        f = lambda x: dphi_funcs[i](x) * dphi_funcs[j](x)
        fhat = lambda xi: f(0.5*xi+0.5)
        for k,p in enumerate(gl2_pts):
            ke_gl2[i,j] += 0.5*fhat(p)*gl2_wts[k]
        for k,p in enumerate(gl4_pts):
            ke_gl4[i,j] += 0.5*fhat(p)*gl4_wts[k]

print(ke_gl2)
print(ke_gl4)