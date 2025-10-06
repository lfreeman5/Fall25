import numpy as np
import matplotlib.pyplot as plt

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
        print(f'Polynomial order {N} for i={i} prediction is {x_i}')
        x_i0 = x_i-1e-3 # Initializing point for secant method

        for _ in range(max_iters):
            print(x_i)
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

### Below this point code written by Copilot ###

def compare_gll_methods(N):
    """
    Compare GLL roots/weights from custom function and numpy-based GLL.
    """
    # Custom GLL
    my_points, my_weights = p_n_roots_weights(N)
    print(f'Custom GLL points:\n{my_points}')
    print(f'Custom GLL weights:\n{my_weights}')

    # Numpy-based GLL
    from numpy.polynomial.legendre import legder, legroots, legval
    if N < 1:
        raise ValueError("Order N must be >= 1")
    Pn_coeff = [0]*N + [1]
    dPn_coeff = legder(Pn_coeff)
    interior_nodes = legroots(dPn_coeff)
    np_points = np.concatenate(([-1.0], interior_nodes, [1.0]))
    np_weights = 2.0 / (N*(N+1) * legval(np_points, Pn_coeff)**2)
    print(f'Numpy-based GLL points:\n{np_points}')
    print(f'Numpy-based GLL weights:\n{np_weights}')

    # Compare endpoints and interior nodes
    print("\nComparison (first and last points should be -1 and 1 for GLL):")
    print(f"Custom GLL endpoints: {my_points[0]}, {my_points[-1]}")
    print(f"Numpy-based GLL endpoints: {np_points[0]}, {np_points[-1]}")

    print("\nInterior nodes (custom GLL):", my_points[1:-1])
    print("Interior nodes (numpy-based GLL):", np_points[1:-1])

    print("\nCustom GLL weights:", my_weights)
    print("Numpy-based GLL weights:", np_weights)

    print(my_points-np_points)
    print(my_weights-np_weights)

def check_legendre_polynomial():
    xi = np.random.uniform(-1, 1)
    N = np.random.randint(0, 11)
    val = p_n(xi,N)
    c = np.zeros(N+1)
    c[-1] = 1
    np_val = np.polynomial.legendre.legval(xi, c)
    print(f'This evaluation: {val} vs np: {np_val} and error: {abs(val-np_val)}')
    
if __name__ == '__main__':
    check_legendre_polynomial()
    # Compare GLL roots/weights with numpy-based implementation
    N = 12
    compare_gll_methods(N)