import sympy as sp

# symbols
x, c1, c2 = sp.symbols('x c1 c2')

# trial function
U_N = 1 + c1*x + c2*(x**2 - x)

# derivative
U_Np = sp.diff(U_N, x)

# functional
I = sp.Rational(3,2) * sp.integrate(U_Np**2 * U_N, (x, 0, 1)) - 4 * sp.integrate(U_N, (x, 0, 1))
print("Functional I:")
print(sp.simplify(I))

# partial derivatives
dI_dc1 = sp.diff(I, c1)
dI_dc2 = sp.diff(I, c2)

print("\ndI/dc1:")
print(sp.simplify(dI_dc1))
print("\ndI/dc2:")
print(sp.simplify(dI_dc2))

# solve stationary conditions
sol = sp.solve([dI_dc1, dI_dc2], [c1, c2], dict=True)
print("\nSolutions for c1, c2:")
for s in sol:
    print({k: sp.simplify(v) for k,v in s.items()})
