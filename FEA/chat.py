import sympy as sp

x = sp.symbols('x')
c1, c2 = sp.symbols('c1 c2')

# Base function satisfying BCs
u_bc = 1

# Trial functions satisfying homogeneous BCs
phi1 = x**2 - 2*x
phi2 = x**3 - 3*x**2 + 2*x

# Approximate solution
u = u_bc + c1*phi1 + c2*phi2

# Residual of PDE
R = -2*u*sp.diff(u, x, 2) + sp.diff(u, x)**2 - 4

# Galerkin conditions
eq1 = sp.integrate(R*phi1, (x, 0, 1))
eq2 = sp.integrate(R*phi2, (x, 0, 1))

# Solve for coefficients
sol = sp.solve([eq1, eq2], (c1, c2), dict=True)

# Display solutions
for s in sol:
    u_sol = u.subs(s)
    print("u(x) =", u_sol)
    print("u0) =", u.subs(s).subs(x, 0))
    print("u'(1) =", sp.diff(u, x).subs(s).subs(x, 1))  # should be 0
