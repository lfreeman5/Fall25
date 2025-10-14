import numpy as np
import matplotlib.pyplot as plt

def generate_spd_matrix(n):
    A = np.random.randn(n, n)
    return A @ A.T + n * np.eye(n)

if __name__ == '__main__':
    N=5
    A = generate_spd_matrix(N)
    B = np.zeros_like(A)
    B[-1,-1]=1.
    sz = 100
    modify_NN = np.linspace(-10,10)
    dets = [np.linalg.det(A-m*B) for m in modify_NN]

    # Plot the determinant as a function of m
    plt.figure(figsize=(10, 6))
    plt.plot(modify_NN, dets, 'b-', linewidth=2)
    plt.xlabel('Modification parameter (m)')
    plt.ylabel('det(A - m*B)')
    plt.title('Determinant of Modified Matrix A as a Function of m')
    plt.grid(True, alpha=0.3)
    plt.show()

