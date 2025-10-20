import sympy as sp

x, L, A = sp.symbols('x L A', positive=True)
h=L/2
phi1, phi2 = 1 - x/h, x/h
dphi1, dphi2 = sp.diff(phi1, x), sp.diff(phi2, x)

phiq = [(1-2*x/h)*(1-x/h), 4*x/h*(1-x/h), -x/h*(1-2*x/h)]
dphiq = [sp.diff(phi,x) for phi in phiq]

k12 = sp.integrate(dphi1*dphi2*A, (x, 0, L/2)) + sp.integrate(dphi1*dphi2*2*A, (x, L/2, L))
k21 = k12
k11 = sp.integrate(dphi1*dphi1*A, (x, 0, L/2)) + sp.integrate(dphi1*dphi1*2*A, (x, L/2, L))
k22 = sp.integrate(dphi2*dphi2*A, (x, 0, L/2)) + sp.integrate(dphi2*dphi2*2*A, (x, L/2, L))
K_1a = [[k11, k12],[k21, k22]]
print(K_1a)

for dp in dphiq:
    print(sp.simplify(dp))

K = sp.zeros(3)
for i in range(3):
    for j in range(3):
        K[i,j] = sp.integrate(dphiq[i]*dphiq[j]*A, (x, 0, L/2)) + sp.integrate(dphiq[i]*dphiq[j]*2*A, (x, L/2, L))

print(sp.simplify(K))

print(sp.integrate(phi1,(x,0,L)))
print(sp.integrate(phi2,(x,0,L)))
print([sp.integrate(phi,(x,0,L)) for phi in phiq])

print(f'Global:')
phis=[phi1,phi2]
dphi = [sp.diff(phi,x) for phi in phis]
K1 = sp.zeros(2)
K2 = sp.zeros(2)
for i in range(2):
    for j in range(2):
        K1[i,j] = sp.integrate(dphi[i]*dphi[j]*A, (x, 0, L/2))
        K2[i,j] = sp.integrate(dphi[i]*dphi[j]*2*A, (x, L/2, L))
print(K1)
print(K2)

print(sp.integrate(phi1,(x,0,L/2)))
print(sp.integrate(phi2,(x,0,L/2)))

