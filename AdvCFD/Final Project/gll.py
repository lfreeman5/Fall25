import numpy as np
import scipy
import gll_utils

def p_n(xi,N):
    '''
        Evaluates the nth legendre polynomial via bonnet formula
    '''
    assert (xi<=1 and xi>=-1), 'ξ must be between -1 and 1 in legendre polynomials'
    p0=1
    p1=xi
    if(N==0): return p0
    if(N==1): return p1
    for n in range(2,N+1):
        pn = ((2*n-1)*xi*p1-(n-1)*p0) / (n)
        p0=p1
        p1=pn
    return pn

def p_prime_n(xi,N):
    '''
        Evaluates the derivative of the nth legendre polynomial
    '''
    return N*(p_n(xi,N-1)-xi*p_n(xi,N))/(1-xi**2.)

def p_n_roots_weights(N,max_iters=100,tol=1e-12):
    '''
        Finds the roots of an order-N GLL integration
        There are N+1 nodes. Two are -1,1. The others are the roots of the derivative of the order-N polynomial.
    '''
    weights = np.zeros(N+1)
    points = np.zeros(N+1)

    for i in range(1,N):
        x_i = (1-3*(N-1)/(8*(N-0)**3.)) *\
            np.cos((4*i+1)/(4*(N)+1) * np.pi) # initial guess at root
        x_i0 = x_i-1e-3 # Initializing point for secant method

        for _ in range(max_iters):
            x1 = x_i - p_prime_n(x_i,N)*(x_i0-x_i) / (p_prime_n(x_i0,N)-p_prime_n(x_i,N))
            if(abs(x1-x_i)<tol):
                points[i] = x1
                weights[i] = 2 / ((N+1)*(N)*p_n(x1,N)**2)
                break
            x_i0=x_i
            x_i=x1
    
    points[0], weights[0] = -1, 2/((N+1)*(N))
    points[-1], weights[-1] = 1, weights[0]
    points, weights = zip(*sorted(zip(points, weights)))  # sort by points
    points, weights = np.array(points), np.array(weights)
    return points, weights

def gll_integrate(f,N):
    '''
    This function integrates f(x) on domain [-1,1] with GLL quadrature
    f is callable with argument x
    N is the order, meaning the integral is calculated with N+1 points
    '''
    pts, wts = p_n_roots_weights(N)
    val = 0
    for i,p in enumerate(pts):
        val = val + wts[i]*f(p)
    return val

def map_function(a,b,f):
    '''
        Maps f(x) on interval [a,b] to g(z) on [-1,1]
        Linear map so T(xi) = (b-a)/2 * xi + (a+b)/2

        returns scaled function based on:
        integral from a to b of f(x) = integral from -1 to 1 of f(T(xi))*(b-a)/2
    '''
    def T(xi):
        return (b-a)/2*xi+(a+b)/2
    def mapped_scaled_f(xi):
        return f(T(xi))*(b-a)/2
    return mapped_scaled_f

def general_integral(a,b,f,N):
    '''
    Integrates f(x) from a to b using GLL
    Uses GLL order N, so N+1 points
    '''
    mapped_scaled_func = map_function(a,b,f)
    return gll_integrate(mapped_scaled_func, N)
    
if __name__ == '__main__':
    N=18
    xi_val = -0.75
    print(f'P_N value for N={N}, xi={xi_val}: My implementation: {p_n(xi_val,N)} np implementation:{gll_utils.eval_pn(xi_val,N)}')

    pts,wts = p_n_roots_weights(N)
    np_pts, np_wts = gll_utils.gll_pts_wts(N)
    print(f'Pts/wts for N={N} integration, my implementation:\n{pts}\n{wts}')
    print(f'Pts/wts for N={N} integration, np implementation:\n{np_pts}\n{np_wts}')
    
    f = lambda x: np.exp(x)*np.sin(x)
    a,b = -2,4
    my_val = general_integral(a,b,f,N)
    np_val = gll_utils.integrate_gll(a,b,f,N)
    print(f'Integral from {a} to {b} of f using: \nMy GLL implementation:{my_val}\nNP GLL: {np_val}\nSciPy Quad: {scipy.integrate.quad(f,a,b,epsabs=1e-12,epsrel=1e-12)[0]}')
    