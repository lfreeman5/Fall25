import sympy as sp

# symbols
x, c1, c2 = sp.symbols('x c1 c2')

# trial function (change based on which)
# U_N = 1-x+c1*(x**2-2*x)+c2*(x**3-x**2) # 2 EBC
U_N = 1+c1*(x**2-2*x)+c2*(x**3+x**2-5*x) # 1 NBC


# derivatives
U_Np = sp.diff(U_N, x)
U_Npp = sp.diff(U_Np, x)

# residual
R = -2*U_N*U_Npp + U_Np**2 - 4
# Only needed for least-squares method
# drc1 = sp.simplify(sp.diff(R,c1))
# drc2 = sp.simplify(sp.diff(R,c2))
# print(f'dR/dc1:  {drc1}')
# print(f'dR/dc2:  {drc2}\n\n')

w_1 = sp.DiracDelta(x-1/3)
w_2 = sp.DiracDelta(x-2/3)



# integrals
I1 = sp.integrate(R*w_1, (x, 0, 1))
I2 = sp.integrate(w_2*R, (x, 0, 1))

# simplify
I1_simplified = sp.simplify(I1)
I2_simplified = sp.simplify(I2)

print(I1_simplified)
print(I2_simplified)

solutions = sp.solve([sp.Eq(I1_simplified,0), sp.Eq(I2_simplified,0)], [c1,c2], dict=True)
for sol in solutions:
    for key in sol:
        sol[key] = float(sol[key])
print(solutions)
