import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
import sympy as sp
pi=np.pi

# Constants
N = 10 # Number of elements - number of nodes is N+1
L = 2 # No longer assumed
h = L/N
k = 1

### This section allows for arbitrary manufactured solution
x = sp.Symbol('x')
t = sp.Symbol('t')
F = sp.sin(sp.pi*x)*sp.exp(sp.sin(sp.pi*t))
f_sym = sp.diff(F, t, 1) - k * sp.diff(F, x, 2)
exact = sp.lambdify((x, t), F, 'numpy')
forcing_func = sp.lambdify((x, t), f_sym, 'numpy')
u_l = lambda t_val: exact(0.0, t_val)
u_r = lambda t_val: exact(L, t_val)

def create_elemental_K():
    K_e=np.array([[1,-1],[-1,1]])*k/h
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
    K = np.zeros((N+1, N+1))
    M = np.zeros((N+1, N+1))
    f = np.zeros(N+1)
    # loop elements e = 1..N. element e spans nodes [e-1, e]
    for e in range(1, N+1):
        n1 = e-1
        n2 = e
        fe = create_elemental_f(e, t_current)   
        K[n1:n2+1, n1:n2+1] += K_e
        M[n1:n2+1, n1:n2+1] += M_e
        f[n1] += fe[0]
        f[n2] += fe[1]

    return K, M, f

def modify_matrices_bcs(A,b,ul,ur):
    '''
    Modifies A and b for LHS and RHS dirichlet boundary conditions 
    '''
    A[0,:] *= 0
    A[:,0] *= 0
    A[-1,:] *= 0
    A[0,0], A[-1,-1] = 1, 1
    b[0] = ul
    b[-1] = ur
    return A, b

def timestep(t0, u0, dt):
    '''
    advances solution in time by 1 step
    uses backwards Euler, evaluating things at n+1
    '''
    K, M, f = assemble_global_K_M_f(t0+dt)
    A = M/dt + K
    b = f + (M@u0)/dt
    A,b = modify_matrices_bcs(A,b,u_l(t0+dt),u_r(t0+dt))
    u1 = np.linalg.solve(A,b)
    return u1

def temporal_solution(tf, dt):
    x = np.linspace(0,L,N+1)
    time = np.arange(dt,tf+dt,dt)
    u0 = exact(x,0.0)
    u = np.zeros((len(time),N+1))
    u[0,:] = u0
    for i,t in enumerate(time):
        u[i+1,:] = timestep(i*dt, u[i,:], dt)
        if(i == len(time)-2):
            break
    return x,time,u

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
    print(f"Problem: u_t = k*u_xx + {f_sym}on 0<x<{L}")

    # Parameters for time integration
    tf = 1.0  # final time
    dt = 0.01 # time step

    # Compute transient solution
    x, time, u = temporal_solution(tf, dt)

    # Plot at each timestep
    x_fine = np.linspace(0, L, 10 * N + 1)
    for i, t in enumerate(np.insert(time, 0, 0.0)):
        plt.ylim((1.1*np.min(u),1.1*np.max(u)))
        plt.clf()
        plt.plot(x, u[i, :], label='Numerical')
        plt.plot(x_fine, exact(x_fine, t), '--', label='Exact')
        plt.title(f"t = {t:.3f}")
        plt.xlabel('x')
        plt.ylabel('u(x, t)')
        plt.legend()
        plt.pause(0.05)
    plt.show()
