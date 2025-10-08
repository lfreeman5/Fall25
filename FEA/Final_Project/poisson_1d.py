import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
pi=np.pi

### This section allows for arbitrary manufactured solution
x = sp.Symbol('x')
F = sp.exp(x)*sp.sin(sp.pi*x)+x-x**2
f = -sp.diff(F,x,2)
exact = sp.lambdify(x,F,'numpy')
forcing_func = sp.lambdify(x,f,'numpy')


L = 2 # No longer assumed
u_l, u_r = exact(0.0), exact(L)
N = 15 # Number of elements - number of nodes is N+1
h = L/N


def create_elemental_K():
    K_e=np.array([[1,-1],[-1,1]])*1/h
    return K_e

def create_elemental_f(element_idx):
    f = np.zeros(2)
    start_x = (element_idx-1)*h
    forcing_function = lambda xbar: forcing_func(xbar+start_x)
    f_phi_1 = lambda x: (1-x/h)*forcing_function(x)
    f_phi_2 = lambda x: (x/h)*forcing_function(x)
    f[0] = quad(f_phi_1, 0, h)[0]
    f[1] = quad(f_phi_2, 0, h)[0]
    return f

def assemble_global_K_f():
    K_e = create_elemental_K()
    K = np.zeros((N+1,N+1))
    f = np.zeros(N+1)
    K[0,:2] = K_e[0]
    f[0] = create_elemental_f(1)[0]
    for i in range(1,N):
        # Create row i
        K[i,i-1] = K_e[1,0]
        K[i,i+1] = K_e[0,1]
        K[i,i] = K_e[0,0] + K_e[1,1]
        f[i] = create_elemental_f(i)[1] + create_elemental_f(i+1)[0] # Maybe wrong indices?
    K[-1,-2:] = K_e[-1]
    f[-1] = create_elemental_f(N)[1]
    return K, f

def restrict_global_K_f(K,f,ul,ur):
    # Applies double-dirichlet boundaries by making K into (N-1)x(N-1)
    f[1] += 1/h*ul
    f[-2] += 1/h*ur

    return K[1:-1,1:-1], f[1:-1]

def create_u_N(u,ul,ur):
    '''
    Creates a piecewise function that represents the approximate solution
    '''
    u = np.concatenate(([ul],u,[ur]))
    def u_N(x):
        xbar = x%h
        idx = int(np.floor((x-xbar)/h))
        return u[idx]*(1-xbar/h) + u[idx+1]*xbar/h
    return u_N

def print_matrix(label,mat):
    arr = np.array(mat)
    print(f'{label}:')
    for row in arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else 1):
        print(" ".join(f"{val:12.6g}" for val in row))

if __name__ == '__main__':
    print(f'{N} Elements, Element size: {h}')
    print(f"Exact solution: u(x) = {F}")
    print(f"Problem: -u''(x) = {f} on 0<x<{L}")
    K,f = assemble_global_K_f()
    K,f = restrict_global_K_f(K,f,u_l,u_r)
    u = np.linalg.solve(K,f)
    u_N = create_u_N(u,u_l,u_r)
    
    # Plot the solution
    x_plot = np.linspace(0, L, 200,endpoint=False) 
    u_N_vals = [u_N(x) for x in x_plot]
    u_exact_vals = exact(x_plot)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_N_vals, 'b-', label=f'Approximate solution (N={N})', linewidth=2)
    plt.plot(x_plot, u_exact_vals, 'r--', label='Exact solution', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title('1D Poisson Equation Solution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()