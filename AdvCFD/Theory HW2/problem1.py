import matplotlib.pyplot as plt
import numpy as np

L = 1
c = 1000
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

def compare_solutions(n, Pe_desired):
    # Produces 2-norm error estimate of solution at given n, Pe
    global c; c = Pe_desired
    x_fd, u_fd = fd_solution(n)
    u_exact = exact_solution(x_fd)
    #    return np.linalg.norm(u_exact-u_fd) / np.linalg.norm(u_exact) / n adding n here gives linear convergence
    return np.linalg.norm(u_exact-u_fd) / np.linalg.norm(u_exact)


def plot_solutions(n_values):
    x_exact = np.linspace(0, L, 200)
    u_exact = exact_solution(x_exact)
    plt.figure(figsize=(8,6))
    plt.plot(x_exact, u_exact, 'k-', label='Exact Solution')
    for n in n_values:
        x_fd, u_fd = fd_solution(n)
        plt.plot(x_fd, u_fd, marker='o', label=f'FD Solution ($n={n}$, $Pe_g={c*L/n/nu}$)')
    plt.xlabel('$x$')
    plt.ylabel('$u(x)$')
    plt.title(f'$Pe = {Pe}$')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_error_vs_n(n_values, Pe_values):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    axes = axes.flatten()
    for idx, Pe_val in enumerate(Pe_values):
        errors = []
        n_list = []
        for n in n_values:
            n_int = int(round(n))
            if n_int < 2:
                continue  # skip invalid n
            err = compare_solutions(n_int, Pe_val)
            errors.append(err)
            n_list.append(n_int)
        axes[idx].loglog(n_list, errors, marker='o', label=f'$Pe={Pe_val}$')
        axes[idx].set_xlabel('$n$')
        axes[idx].set_ylabel('$\epsilon$')
        axes[idx].set_title(f'$Pe={Pe_val}$')
        axes[idx].grid(True, which="both", ls="--")
        axes[idx].legend()
    plt.tight_layout()
    plt.show()

def main():
    n_values = [40]
    plot_solutions(n_values)
    n_values = np.geomspace(4,1024,9)
    Pe_vals = [1,10,100,1000]
    plot_error_vs_n(n_values, Pe_vals)


if __name__ == "__main__":
    main()