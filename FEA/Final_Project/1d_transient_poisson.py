import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
pi=np.pi

# User parameters
element_type = 'linear'  # 'linear' or 'quadratic'
N = 25  # Number of elements
L = 4
k = 1
h = L/N

### Manufactured solution
x = sp.Symbol('x')
t = sp.Symbol('t')
F = sp.sin(sp.pi*x*t)*sp.sin(sp.pi*x)
f_sym = sp.diff(F, t, 1) - k * sp.diff(F, x, 2)
exact = sp.lambdify((x, t), F, 'numpy')
forcing_func = sp.lambdify((x, t), f_sym, 'numpy')
u_l = lambda t_val: exact(0.0, t_val)
u_r = lambda t_val: exact(L, t_val)

def create_elemental_K_linear():
    return np.array([[1,-1],[-1,1]])*k/h

def create_elemental_M_linear():
    return np.array([[2,1],[1,2]])*(h/6)

def create_elemental_K_quadratic():
    # (1/3h) * [[7, -8, 1], [-8, 16, -8], [1, -8, 7]]
    return np.array([[7,-8,1],[-8,16,-8],[1,-8,7]]) * (k/(3*h))

def create_elemental_M_quadratic():
    # (h/30) * [[4,2,-1],[2,16,2],[-1,2,4]]
    return np.array([[4,2,-1],[2,16,2],[-1,2,4]]) * (h/30)

def create_elemental_f_linear(element_idx, t_current):
    f = np.zeros(2)
    start_x = (element_idx-1)*h
    forcing_function = lambda xbar: forcing_func(xbar+start_x, t_current)
    f_phi_1 = lambda x: (1-x/h)*forcing_function(x)
    f_phi_2 = lambda x: (x/h)*forcing_function(x)
    f[0] = quad(f_phi_1, 0, h)[0]
    f[1] = quad(f_phi_2, 0, h)[0]
    return f

def create_elemental_f_quadratic(element_idx, t_current):
    f = np.zeros(3)
    start_x = (element_idx-1)*h
    forcing_function = lambda xbar: forcing_func(xbar+start_x, t_current)
    # Standard quadratic shape functions on [0, h]
    phi1 = lambda x: 2*((x/h)-0.5)*((x/h)-1)
    phi2 = lambda x: 4*(x/h)*(1-x/h)
    phi3 = lambda x: 2*(x/h)*((x/h)-0.5)
    f[0] = quad(lambda x: phi1(x)*forcing_function(x), 0, h)[0]
    f[1] = quad(lambda x: phi2(x)*forcing_function(x), 0, h)[0]
    f[2] = quad(lambda x: phi3(x)*forcing_function(x), 0, h)[0]
    return f

def assemble_global_K_M_f(t_current, element_type='linear'):
    if element_type == 'linear':
        K_e = create_elemental_K_linear()
        M_e = create_elemental_M_linear()
        num_nodes = N + 1
        K = np.zeros((num_nodes, num_nodes))
        M = np.zeros((num_nodes, num_nodes))
        f = np.zeros(num_nodes)
        for e in range(1, N+1):
            n1 = e-1
            n2 = e
            fe = create_elemental_f_linear(e, t_current)
            K[n1:n2+1, n1:n2+1] += K_e
            M[n1:n2+1, n1:n2+1] += M_e
            f[n1] += fe[0]
            f[n2] += fe[1]
        return K, M, f
    elif element_type == 'quadratic':
        K_e = create_elemental_K_quadratic()
        M_e = create_elemental_M_quadratic()
        num_nodes = 2*N + 1
        K = np.zeros((num_nodes, num_nodes))
        M = np.zeros((num_nodes, num_nodes))
        f = np.zeros(num_nodes)
        for e in range(N):
            idx = 2*e
            fe = create_elemental_f_quadratic(e+1, t_current)
            K[idx:idx+3, idx:idx+3] += K_e
            M[idx:idx+3, idx:idx+3] += M_e
            f[idx:idx+3] += fe
        return K, M, f
    else:
        raise ValueError("element_type must be 'linear' or 'quadratic'")

def modify_matrices_bcs(A, b, ul, ur):
    A[0,:] = 0
    A[:,0] = 0
    A[-1,:] = 0
    A[:,-1] = 0
    A[0,0], A[-1,-1] = 1, 1
    b[0] = ul
    b[-1] = ur
    return A, b

def timestep(t0, u0, dt, element_type='linear'):
    K, M, f = assemble_global_K_M_f(t0+dt, element_type)
    A = M/dt + K
    b = f + (M@u0)/dt
    A, b = modify_matrices_bcs(A, b, u_l(t0+dt), u_r(t0+dt))
    u1 = np.linalg.solve(A, b)
    return u1

def temporal_solution(tf, dt, element_type='linear'):
    if element_type == 'linear':
        num_nodes = N + 1
        x = np.linspace(0, L, num_nodes)
    else:
        num_nodes = 2*N + 1
        x = np.linspace(0, L, num_nodes)
    time = np.arange(dt, tf+dt, dt)
    u0 = exact(x, 0.0)
    u = np.zeros((len(time)+1, num_nodes))
    u[0,:] = u0
    for i, t in enumerate(time):
        u[i+1,:] = timestep(i*dt, u[i,:], dt, element_type)
    return x, time, u

def create_u_N_quadratic(u, ul, ur):
    # u: solution vector (excluding boundaries)
    u_full = np.concatenate(([ul], u, [ur]))
    def u_N(x):
        if x <= 0:
            return ul
        if x >= L:
            return ur
        e = int(np.floor(x / h))
        xi = (x - e * h) / h  # local coordinate in [0,1]
        i0 = 2 * e
        # Standard quadratic shape functions on [0,1]
        phi1 = 2*(xi-0.5)*(xi-1)
        phi2 = 4*xi*(1-xi)
        phi3 = 2*xi*(xi-0.5)
        return (
            u_full[i0]   * phi1 +
            u_full[i0+1] * phi2 +
            u_full[i0+2] * phi3
        )
    return u_N

if __name__ == '__main__':
    print(f'{N} Elements, Element size: {h}')
    print(f"Exact solution: u(x) = {F}")
    print(f"Problem: u_t = k*u_xx + {f_sym} on 0<x<{L}")

    tf = 0.9
    dt = 0.01

    # Compute transient solution
    x, time, u = temporal_solution(tf, dt, element_type)

    # Animation
    x_fine = np.linspace(0, L, 10 * N + 1)
    if element_type == 'linear':
        for i, t in enumerate(np.insert(time, 0, 0.0)):
            plt.clf()
            plt.plot(x, u[i, :], 'bo-', label='Numerical (nodes)')
            plt.plot(x_fine, exact(x_fine, t), 'r--', label='Exact')
            plt.title(f"t = {t:.3f} (Linear)")
            plt.xlabel('x')
            plt.ylabel('u(x, t)')
            plt.legend()
            plt.pause(0.05)
        plt.show()
    elif element_type == 'quadratic':
        for i, t in enumerate(np.insert(time, 0, 0.0)):
            u_N_quad = create_u_N_quadratic(u[i, 1:-1], u[i, 0], u[i, -1])
            u_quad_vals = [u_N_quad(xi) for xi in x_fine]
            plt.clf()
            plt.plot(x_fine, u_quad_vals, 'g-', label='Numerical (quadratic)')
            plt.plot(x_fine, exact(x_fine, t), 'r--', label='Exact')
            plt.title(f"t = {t:.3f} (Quadratic)")
            plt.xlabel('x')
            plt.ylabel('u(x, t)')
            plt.legend()
            plt.pause(0.05)
        plt.show()

    # Plot only the final timestep
    t_final = time[-1]
    u_final = u[-1, :]
    u_exact_final = exact(x_fine, t_final)

    plt.figure(figsize=(10, 6))
    if element_type == 'linear':
        plt.plot(x, u_final, 'b-', label='Numerical (linear)')
        plt.plot(x_fine, u_exact_final, 'r--', label='Exact')
    elif element_type == 'quadratic':
        u_N_quad = create_u_N_quadratic(u_final[1:-1], u_final[0], u_final[-1])
        u_quad_vals = [u_N_quad(xi) for xi in x_fine]
        plt.plot(x_fine, u_quad_vals, 'g-', label='Numerical (quadratic)')
        plt.plot(x_fine, u_exact_final, 'r--', label='Exact')
    plt.xlabel('x')
    plt.ylabel('u(x, t_final)')
    # plt.title(f'Solution at final time t={t_final:.3f} ({element_type.capitalize()} elements)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
