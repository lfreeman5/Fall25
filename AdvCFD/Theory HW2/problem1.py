import matplotlib.pyplot as plt
import numpy as np

L = 1
c = 1
nu = 1
Pe = c*L/nu
u_L, u_R = 0, 0 # Homogenous Dirichlet conditions are assumed

def exact_solution(x):
    z = x/L
    return L/c * (z - (np.exp(Pe * (z-1)) - np.exp(-Pe)) / (1-np.exp(-Pe)))

def fd_solution(n):
    N = n + 1
    dx = L / n
    alpha = nu / (dx**2)
    beta = c/(2*dx)
    A = np.zeros((N-2,N-2)) # Only solves for internal points
    for i in range(1,N-3):
        A[i,i-1] = -alpha - beta
        A[i,i] = 2*alpha
        A[i,i+1] = -alpha + beta
    A[0,0] = 2*alpha
    A[0,1] = -alpha + beta
    A[-1,-1] = 2*alpha
    A[-1,-2] = -alpha - beta
    f = np.ones(N-2)
    u_int = np.linalg.solve(A,f)
    u_full = np.hstack(([u_L], u_int, [u_R]))
    x = np.linspace(0,L,N)
    return x, u_full

def plot_solutions(n_values):
    x_exact = np.linspace(0, L, 200)
    u_exact = exact_solution(x_exact)
    plt.figure(figsize=(8,6))
    plt.plot(x_exact, u_exact, 'k-', label='Exact Solution')
    for n in n_values:
        x_fd, u_fd = fd_solution(n)
        plt.plot(x_fd, u_fd, marker='o', label=f'FD Solution (n={n})')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.title(f'Pe = {Pe}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    n_values = [5, 10, 20, 50]
    plot_solutions(n_values)

if __name__ == "__main__":
    main()