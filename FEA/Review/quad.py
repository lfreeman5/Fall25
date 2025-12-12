import numpy as np
from scipy.integrate import newton_cotes
from scipy.integrate import quad

def nc_quad(a,b,f,N):
    dx = (b-a)/(N-1)
    val = 0.
    weights, _ = newton_cotes(N-1)
    weights/=(N-1)
    print(f'NC weights:{weights}')
    for i in range(N):
        val += weights[i]*f(a+i*dx)
    return val*(b-a)

def gl_quad(a,b,f,N): 
    points, weights = np.polynomial.legendre.leggauss(N)
    print(f'GL weights: {weights}')
    val = 0.
    for i,p in enumerate(points):
        x_i = a+(b-a)/2 * (1+p)
        val += f(x_i)*weights[i]
    return val*(b-a)/2.

if __name__ == '__main__':
    f = lambda x: np.exp(x)
    a=2
    b=5
    N=6
    nc_val = nc_quad(a,b,f,N)
    gl_val = gl_quad(a,b,f,N)
    quad_val, _ = quad(f, a, b)
    print(f"Newton-Cotes: {nc_val}")
    print(f"Gauss-Legendre: {gl_val}")
    print(f"scipy quad: {quad_val}")
    print(f"NC error: {abs(nc_val - quad_val)}")
    print(f"GL error: {abs(gl_val - quad_val)}")