import sympy as sp

# symbols
x, c1, c2 = sp.symbols('x c1 c2')

# trial function
U_N = 1 - x + c1*(x**2 - x) + c2*(x**3 - x**2)

# derivatives
U_Np = sp.diff(U_N, x)
U_Npp = sp.diff(U_Np, x)

# residual
R = -2*U_N*U_Npp + U_Np**2 - 4

# integrals
I1 = sp.integrate(R, (x, 0, 1))
I2 = sp.integrate(x*R, (x, 0, 1))

# simplify
I1_simplified = sp.simplify(I1)
I2_simplified = sp.simplify(I2)

print(I1_simplified)
print(I2_simplified)

