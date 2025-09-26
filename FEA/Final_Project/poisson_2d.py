import numpy as np
import matplotlib.pyplot as plt

pi=np.pi

def analytical_solution(x,y):
    return np.sin(pi*x)*np.sin(pi*y)

def plot_u_analytical():
    """Plot u(x,y) as a color plot on domain 0<x<1, 0<y<1"""
    # Create meshgrid
    x = np.linspace(0, 1, 100)  # Start slightly above 0 to avoid potential issues
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    
    # Evaluate solution
    U = analytical_solution(X, Y)
    
    # Create color plot
    plt.figure(figsize=(7, 6))
    contour = plt.contourf(X, Y, U, levels=50, cmap='jet')
    plt.colorbar(contour, label='u(x,y)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True, alpha=0.3)
    plt.show()

if __name__ == "__main__":
    plot_u_analytical()