import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
pi=np.pi

### This section allows for arbitrary manufactured solution
x = sp.Symbol('x')
F = sp.sin(sp.pi*x)
f = -sp.diff(F,x,2)
exact = sp.lambdify(x,F,'numpy')
forcing_func = sp.lambdify(x,f,'numpy')


L = 1 # No longer assumed to be 1
u_l, u_r = exact(0.0), exact(L)
N = 8 # Number of elements - number of nodes is N+1
h = L/N


def create_elemental_K_linear():
    K_e=np.array([[1,-1],[-1,1]])*1/h
    return K_e

def create_elemental_K_quadratic():
    K_e = np.array([[7,-8,1],
                    [-8,16,-8],
                    [1,-8,7]])*1/(3*h)
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

def create_elemental_f_quadratic(element_idx):
    f = np.zeros(3)
    start_x = (element_idx-1)*h
    forcing_function = lambda xbar: forcing_func(xbar+start_x)
    f_phi_1 = lambda x: (1-2*x/h)*(1-x/h)*forcing_function(x)
    f_phi_2 = lambda x: 4*(x/h)*(1-x/h)*forcing_function(x)
    f_phi_3 = lambda x: -1*(x/h)*(1-2*x/h)*forcing_function(x)
    f[0] = quad(f_phi_1, 0, h)[0]
    f[1] = quad(f_phi_2, 0, h)[0]
    f[2] = quad(f_phi_3, 0, h)[0]
    return f

def assemble_global_K_f(element_type='linear'):
    if element_type == 'linear':
        K_e = create_elemental_K_linear()
        num_nodes = N + 1
        K = np.zeros((num_nodes, num_nodes))
        f = np.zeros(num_nodes)
        K[0,:2] = K_e[0]
        f[0] = create_elemental_f(1)[0]
        for i in range(1,N):
            K[i,i-1] = K_e[1,0]
            K[i,i+1] = K_e[0,1]
            K[i,i] = K_e[0,0] + K_e[1,1]
            f[i] = create_elemental_f(i)[1] + create_elemental_f(i+1)[0]
        K[-1,-2:] = K_e[-1]
        f[-1] = create_elemental_f(N)[1]
        return K, f
    elif element_type == 'quadratic':
        K_e = create_elemental_K_quadratic()
        num_nodes = 2*N + 1
        K = np.zeros((num_nodes, num_nodes))
        f = np.zeros(num_nodes)
        for e in range(N):
            idx = 2*e
            K[idx:idx+3, idx:idx+3] += K_e
            f[idx:idx+3] += create_elemental_f_quadratic(e+1)
        return K, f
    else:
        raise ValueError("element_type must be 'linear' or 'quadratic'")

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

def create_u_N_quadratic(u, ul, ur):
    '''
    Creates a piecewise quadratic function that represents the approximate solution
    u: solution vector (excluding boundaries)
    ul, ur: Dirichlet boundary values
    '''
    # u should be of length 2*N-1 (excluding boundaries)
    u_full = np.concatenate(([ul], u, [ur]))
    def u_N(x):
        if x <= 0:
            return ul
        if x >= L:
            return ur
        e = int(np.floor(x / h))
        xi = (x - e * h) / h  # local coordinate in [0,1]
        i0 = 2 * e
        # Quadratic shape functions on [0, h]:
        phi1 = (1 - 2*xi) * (1 - xi)
        phi2 = 4 * xi * (1 - xi)
        phi3 = -xi * (1 - 2*xi)
        return (
            u_full[i0]   * phi1 +
            u_full[i0+1] * phi2 +
            u_full[i0+2] * phi3
        )
    return u_N

def print_matrix(label,mat):
    arr = np.array(mat)
    print(f'{label}:')
    for row in arr.reshape(-1, arr.shape[-1] if arr.ndim > 1 else 1):
        print(" ".join(f"{val:12.6g}" for val in row))

def calc_residual(u_exact, u_N):
    diff = lambda x: (u_exact(x)-u_N(x))**2
    res = (quad(diff, 0, L, limit=400, epsabs=1e-6, epsrel=1e-6)[0])**0.5
    return res


if __name__ == '__main__':
    print(f'{N} Elements, Element size: {h}')
    print(f"Exact solution: u(x) = {F}")
    print(f"Problem: -u''(x) = {f} on 0<x<{L}")

    # Linear solution
    K_lin, f_lin = assemble_global_K_f('linear')
    K_lin, f_lin = restrict_global_K_f(K_lin, f_lin, u_l, u_r)
    u_lin = np.linalg.solve(K_lin, f_lin)
    u_N_lin = create_u_N(u_lin, u_l, u_r)

    # Plot linear solution
    x_plot = np.linspace(0, L, 200, endpoint=False)
    u_N_lin_vals = [u_N_lin(x) for x in x_plot]
    u_exact_vals = exact(x_plot)
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_N_lin_vals, 'b-', label=f'Linear FEM (N={N})', linewidth=2)
    plt.plot(x_plot, u_exact_vals, 'r--', label='Exact solution', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    # plt.title('1D Poisson Equation Solution (Linear Elements)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Quadratic solution
    K_quad, f_quad = assemble_global_K_f('quadratic')
    # For quadratic, boundaries are at indices 0 and -1
    K_quad, f_quad = restrict_global_K_f(K_quad, f_quad, u_l, u_r)
    u_quad = np.linalg.solve(K_quad, f_quad)
    u_N_quad = create_u_N_quadratic(u_quad, u_l, u_r)

    # Plot quadratic solution
    x_plot = np.linspace(0, L, 400, endpoint=False)
    u_N_quad_vals = [u_N_quad(x) for x in x_plot]
    u_exact_vals = exact(x_plot)
    plt.figure(figsize=(10, 6))
    plt.plot(x_plot, u_N_quad_vals, 'g-', label=f'Quadratic FEM (N={N})', linewidth=2)
    plt.plot(x_plot, u_exact_vals, 'r--', label='Exact solution', linewidth=2)
    plt.xlabel('x')
    plt.ylabel('u(x)')
    # plt.title('1D Poisson Equation Solution (Quadratic Elements)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # Convergence study for both linear and quadratic
    residual_Ns = [5,10,15,20,25,30,40,55]
    residuals_lin = []
    residuals_quad = []
    for rN in residual_Ns:
        # Linear
        N = rN
        h = L/N
        K, f = assemble_global_K_f('linear')
        K, f = restrict_global_K_f(K, f, u_l, u_r)
        u = np.linalg.solve(K, f)
        u_N = create_u_N(u, u_l, u_r)
        residuals_lin.append(calc_residual(exact, u_N))
        # Quadratic
        Nq = rN
        hq = L/Nq
        Kq, fq = assemble_global_K_f('quadratic')
        Kq, fq = restrict_global_K_f(Kq, fq, u_l, u_r)
        uq = np.linalg.solve(Kq, fq)
        u_Nq = create_u_N_quadratic(uq, u_l, u_r)
        residuals_quad.append(calc_residual(exact, u_Nq))

    plt.figure(figsize=(8,5))
    plt.loglog(residual_Ns, residuals_lin, 'o-', linewidth=2, markersize=5, label='Linear')
    plt.loglog(residual_Ns, residuals_quad, 's-', linewidth=2, markersize=5, label='Quadratic')
    plt.xlabel('$N$')
    plt.ylabel('$L2$ residual $||u - u_N||$')
    plt.legend()
    plt.grid(True, which='both', alpha=0.3)
    # plt.title('Convergence of FEM: Linear vs Quadratic')
    plt.show()