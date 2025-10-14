import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
pi=np.pi

# Constants
N = 135 # Number of elements - number of nodes is N+1
L = 1 # No longer assumed
h = L/N
k = 1

### This section allows for arbitrary manufactured solution
x = sp.Symbol('x')
t = sp.Symbol('t')
F = x**2 / (t+1)
f = sp.diff(F, t, 1) - k * sp.diff(F, x, 2)
exact = sp.lambdify((x, t), F, 'numpy')
forcing_func = sp.lambdify((x, t), f, 'numpy')
u_l = lambda t_val: exact(0.0, t_val)
u_r = lambda t_val: exact(L, t_val)



def create_elemental_K():
    K_e=np.array([[1,-1],[-1,1]])*1/h
    return K_e

def create_elemental_M():
    M_e=np.array([[2,1],[1,2]])*(h/6)
    return M_e

def create_elemental_f(element_idx, t_current):
    f = np.zeros(2)
    start_x = (element_idx-1)*h
    forcing_function = lambda xbar: forcing_func(xbar+start_x,t_current)
    f_phi_1 = lambda x: (1-x/h)*forcing_function(x)
    f_phi_2 = lambda x: (x/h)*forcing_function(x)
    f[0] = quad(f_phi_1, 0, h)[0]
    f[1] = quad(f_phi_2, 0, h)[0]
    return f

def assemble_global_K_M_f(t_current):
    K_e = create_elemental_K()
    M_e = create_elemental_M()
    K = np.zeros((N+1,N+1))
    M = np.zeros((N+1,N+1))
    f = np.zeros(N+1)
    K[0,:2], M[0,:2] = M_e[0]
    f[0] = create_elemental_f(1, t_current)[0]
    for i in range(1,N):
        # Create row i
        K[i,i-1], M[i,i-1] = K_e[1,0], M_e[1,0]
        K[i,i+1], M[i,i+1] = K_e[0,1], M_e[0,1]
        K[i,i], M[i,i] = K_e[0,0] + K_e[1,1], M_e[0,0] + M_e[1,1]
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

def calc_residual(u_exact, u_N):
    diff = lambda x: (u_exact(x)-u_N(x))**2
    res = (quad(diff,0,L,limit=400)[0])**0.5
    return res


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

    residual_Ns = [5,10,15,20,25,30,40,55]
    residuals = []
    for rN in residual_Ns:
        N = rN
        h = L/N
        K,f = assemble_global_K_f()
        K,f = restrict_global_K_f(K,f,u_l,u_r)
        u = np.linalg.solve(K,f)
        u_N = create_u_N(u,u_l,u_r)
        residuals.append(calc_residual(exact,u_N))

    plt.figure(figsize=(8,5))
    plt.loglog(residual_Ns, residuals, 'o-', linewidth=2, markersize=5)
    plt.xlabel('$N$')
    plt.ylabel('$L2$ residual $||u - u_N||$')
    plt.grid(True, which='both', alpha=0.3)
    plt.show()