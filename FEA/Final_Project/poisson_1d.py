import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
pi=np.pi

def exact(x):
    return np.sin(pi*x)

L = 1 # Assumed in derivation of manufactured solution
u_l, u_r = 0, 0
N = 50 # Number of elements - number of nodes is N+1
h = L/N


def create_elemental_K():
    K_e=np.array([[1,-1],[-1,1]])*1/h
    return K_e

def create_elemental_f(x_a):
    f = np.zeros(2)
    forcing_function = lambda xbar: pi**2 * np.sin(pi*(xbar+x_a)) # we want the function to go from 0 to h
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
    f[0] = create_elemental_f(0)[0]
    for i in range(1,N):
        # Create row i
        K[i,i-1] = K_e[1,0]
        K[i,i+1] = K_e[0,1]
        K[i,i] = K_e[0,0] + K_e[1,1]
        f[i] = create_elemental_f(h*(i-1))[1] + create_elemental_f(h*(i))[0] # Maybe wrong indices?
    K[-1,-2:] = K_e[-1]
    f[-1] = create_elemental_f((N-1)*h)[1]
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
        idx = int(np.round((x-xbar)/h))
        return u[idx]*(1-xbar)/h + u[idx+1]*xbar/h
    return u_N

if __name__ == '__main__':
    K,f = assemble_global_K_f()
    K,f = restrict_global_K_f(K,f,u_l,u_r)
    u = np.linalg.solve(K,f)
    u_N = create_u_N(u,u_l,u_r)
    
    # Plot the solution
    x_plot = np.linspace(0, L, 1000)
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