import numpy as np
import matplotlib.pyplot as plt

def lagrange_polynomial(x_nodes, i, x):
    """
    Compute the i-th Lagrange polynomial L_i(x) at points x.
    x_nodes: array of interpolation nodes
    i: index of the Lagrange polynomial
    x: points where the polynomial is evaluated
    """
    L = np.ones_like(x, dtype=float)
    for j, xj in enumerate(x_nodes):
        if j != i:
            L *= (x - xj) / (x_nodes[i] - xj)
    return L

# Example: nodes and plotting
x_nodes = np.array([-1, -0.85, 0, 0.85, 1])  # interpolation nodes
x_plot = np.linspace(-1.2, 1.2, 500)      # fine grid for plotting

plt.figure(figsize=(8,5))

# Plot each Lagrange polynomial
for i in range(len(x_nodes)):
    L_i = lagrange_polynomial(x_nodes, i, x_plot)
    plt.plot(x_plot, L_i, label=f'L_{i}(x)')
    plt.plot(x_nodes[i], 1, 'ko')  # mark the node itself

plt.title('Lagrange Polynomials')
plt.xlabel('x')
plt.ylabel('L_i(x)')
plt.grid(True)
plt.legend()
plt.show()
