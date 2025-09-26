import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

x, c1, c2 = sp.symbols('x c1 c2')
U_N = 1-x+c1*(x**2-x)+c2*(x**3-x**2)
U_Np = sp.diff(U_N, x)
U_Npp = sp.diff(U_Np, x)
R = -2*U_N*U_Npp + U_Np**2 - 4

def calc_c1c2(w1,w2):
    I1 = sp.integrate(R*w1, (x, 0, 1))
    I2 = sp.integrate(w2*R, (x, 0, 1))
    I1_simplified = sp.simplify(I1)
    I2_simplified = sp.simplify(I2)
    solutions = sp.solve([sp.Eq(I1_simplified,0), sp.Eq(I2_simplified,0)], [c1,c2], dict=True)
    for sol in solutions:
        for key in sol:
            sol[key] = float(sol[key])
    print(solutions)
    return solutions

def choose_c1c2(result):
    min_error = float('inf')
    best_c1, best_c2 = None, None
    
    for sol in result:
        c1_val = sol[c1]
        c2_val = sol[c2]
        
        # Substitute c1, c2 values into U_N
        U_N_sub = U_N.subs([(c1, c1_val), (c2, c2_val)])
        
        # Calculate error function: exact(x) - U_N
        error_func = (1 - x**2) - U_N_sub
        
        # Calculate L2 norm: sqrt(integral of error^2 from 0 to 1)
        l2_norm_squared = sp.integrate(error_func**2, (x, 0, 1))
        l2_norm = sp.sqrt(l2_norm_squared)
        
        # Convert to float for comparison
        l2_norm_val = float(l2_norm)
        
        if l2_norm_val < min_error:
            min_error = l2_norm_val
            best_c1, best_c2 = c1_val, c2_val
    
    return best_c1, best_c2

def exact(x):
    return 1-x**2

if __name__ == "__main__":
    pg_w1 = 1
    pg_w2 = x
    g_w1 = x**2-2*x
    g_w2 = x**3-x**2
    ls_w1 = sp.simplify(sp.diff(R,c1))
    ls_w2 = sp.simplify(sp.diff(R,c2))
    co_w1 = sp.DiracDelta(x-1/3)
    co_w2 = sp.DiracDelta(x-2/3)
    
    # Calculate best c1, c2 for each weight function pair
    weight_functions = [
        ("Petrov-Galerkin", pg_w1, pg_w2),
        ("Galerkin", g_w1, g_w2),
        ("Least Squares", ls_w1, ls_w2),
        ("Collocation", co_w1, co_w2)
    ]
    
    solutions = {}
    x_vals = np.linspace(0, 1, 100)
    
    plt.figure(figsize=(10, 6))
    
    for name, w1, w2 in weight_functions:
        try:
            result = calc_c1c2(w1, w2)
            best_c1, best_c2 = choose_c1c2(result)
            solutions[name] = (best_c1, best_c2)
            
            # Create numerical function for plotting
            U_N_best = U_N.subs([(c1, best_c1), (c2, best_c2)])
            U_N_func = sp.lambdify(x, U_N_best, 'numpy')
            
            plt.plot(x_vals, U_N_func(x_vals), label=f'{name} (c1={best_c1:.4f}, c2={best_c2:.4f})')
            print(f"{name}: c1={best_c1:.6f}, c2={best_c2:.6f}")
            
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    # Plot exact solution
    exact_vals = 1 - x_vals**2
    plt.plot(x_vals, exact_vals, 'k--', linewidth=2, label='Exact Solution')
    
    plt.xlabel('x')
    plt.ylabel('U(x)')
    plt.title('Approximate Solutions vs Exact Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

