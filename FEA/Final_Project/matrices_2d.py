import numpy as np
import sympy as sp
from scipy.integrate import dblquad, quad

def create_mass_stiffness_matrices(alpha, dx, dy):

    x,y,a,b = sp.symbols('x y a b')
    phis = [(1-x/a)*(1-y/b),
            (x/a)*(1-y/b),
            (x/a)*(y/b),
            (1-x/a)*(y/b)]
    dxphis = [sp.diff(p,x,1) for p in phis]
    dyphis = [sp.diff(p,y,1) for p in phis]

    m = sp.zeros(4, 4)
    k = sp.zeros(4,4)

    for i in range(4):
        for j in range(4):
            m[i,j] = sp.integrate(phis[i] * phis[j], (x, 0, a), (y, 0, b))
            k[i,j] = sp.integrate(dxphis[i]*dxphis[j] + dyphis[i]*dyphis[j], (x, 0, a), (y, 0, b))

    # Substitute a=dx, b=dy and convert to float numpy arrays
    m_num = np.array(m.subs({'a': dx, 'b': dy})).astype(np.float64)
    k_num = -alpha * np.array(k.subs({'a': dx, 'b': dy})).astype(np.float64)
    return m_num, k_num

def assemble_global_mass_stiffness_matrices(Me, Ke, element_map, N_nodes):
    """
    Assemble global mass (Mg) and stiffness (Kg) matrices.
    """
    Ne = element_map.shape[0]
    Mg = np.zeros((N_nodes, N_nodes))
    Kg = np.zeros((N_nodes, N_nodes))

    for e_num, e in enumerate(element_map):
        for i in range(4):
            I = e[i]
            for j in range(4):
                J = e[j]
                Mg[I, J] += Me[i, j]
                Kg[I, J] += Ke[i, j]

    return Mg, Kg

def assemble_global_force_vector(Fe_arr, element_map, N_nodes):
    """
    Assemble global force vector (Fg).
    """
    Ne = element_map.shape[0]
    Fg = np.zeros(N_nodes)

    for e_num, e in enumerate(element_map):
        for i in range(4):
            I = e[i]
            Fg[I] += Fe_arr[e_num, i]

    return Fg

def eval_f_e(f, xi, yi, xf, yf):
    # Use 2x2 Gauss quadrature for speed
    a = xf - xi
    b = yf - yi
    # Gauss points and weights for [-1,1]
    gp = np.array([-1/np.sqrt(3), 1/np.sqrt(3)])
    w = np.array([1, 1])
    # Shape functions in reference [-1,1]x[-1,1]
    N = [
        lambda xi, eta: 0.25 * (1 - xi) * (1 - eta),
        lambda xi, eta: 0.25 * (1 + xi) * (1 - eta),
        lambda xi, eta: 0.25 * (1 + xi) * (1 + eta),
        lambda xi, eta: 0.25 * (1 - xi) * (1 + eta)
    ]
    f_e = np.zeros(4)
    for i in range(2):
        for j in range(2):
            xi_gp, eta_gp = gp[i], gp[j]
            # Map to physical coordinates
            x = xi + (xi_gp + 1) * a / 2
            y = yi + (eta_gp + 1) * b / 2
            fval = f(x, y)
            for k in range(4):
                f_e[k] += N[k](xi_gp, eta_gp) * fval * w[i] * w[j]
    f_e *= (a * b) / 4.0
    return f_e

def calc_all_fe(f, element_map, xg, yg):
    Ne = element_map.shape[0]
    fe_arr = np.zeros((Ne,4))
    for i,e in enumerate(element_map):
        xi, yi = xg[e[0]], yg[e[0]]
        xf, yf = xg[e[2]], yg[e[2]]
        fe_arr[i] = eval_f_e(f,xi,yi,xf,yf)
    return fe_arr

prev_N = -1
GLxi, GLwi = None, None
def gauss_legendre_integrate_1d(f, a, b, n=4):
    """
    Perform 1D Gauss-Legendre quadrature of function f over [a, b] with n points.
    """
    # Get nodes and weights for [-1, 1]
    global prev_N, GLxi, GLwi
    if(prev_N==n):
        xi, wi = GLxi, GLwi
    else:
        GLxi, GLwi = np.polynomial.legendre.leggauss(n)
        prev_N = n
        xi, wi = GLxi, GLwi
    # Change of interval
    xm = 0.5 * (b + a)
    xr = 0.5 * (b - a)
    s = xm + xr * xi
    return np.sum(wi * f(s)) * xr

def calc_boundary_term(Nx, Ny, dx, dy, xg, yg, q):
    '''
    xg, yg are N=Nx*Ny arrays of global position
    q is the arrays of boundary vectors
    Uses Gauss-Legendre quadrature for speed.
    '''
    [qt,qb,ql,qr] = q
    N = Nx*Ny
    B = np.zeros(N)

    n_gl = 4  # Number of GL points

    # LHS boundary
    if(ql is not None):
        for j in range(Ny):
            idx = j*Nx
            y_i = yg[idx]
            # s in [0, dy]
            integrand_1 = lambda s: ql(y_i+dy-s)*(s/dy)
            integrand_2 = lambda s: ql(y_i-s)*(1-s/dy)
            i1 = gauss_legendre_integrate_1d(integrand_1, 0, dy, n=n_gl)
            i2 = gauss_legendre_integrate_1d(integrand_2, 0, dy, n=n_gl)
            if(j==0):
                B[idx] += i1
            elif(j==Ny-1):
                B[idx] += i2
            else:
                B[idx] = i1+i2
    # RHS boundary
    if(qr is not None):
        for j in range(Ny):
            idx = j*Nx + (Nx-1)
            y_i = yg[idx]
            integrand_1 = lambda s: qr(y_i-dy+s)*(s/dy)
            integrand_2 = lambda s: qr(y_i+s)*(1-s/dy)
            i1 = gauss_legendre_integrate_1d(integrand_1, 0, dy, n=n_gl)
            i2 = gauss_legendre_integrate_1d(integrand_2, 0, dy, n=n_gl)
            if(j==0):
                B[idx] += i2
            elif(j==Ny-1):
                B[idx] += i1
            else:
                B[idx] = i1+i2
    # Bottom boundary
    if(qb is not None):
        for i in range(Nx):
            idx = i
            x_i = xg[idx]
            integrand_1 = lambda s: qb(x_i - dx + s)*(s/dx)
            integrand_2 = lambda s: qb(x_i + s)*(1-s/dx)
            i1 = gauss_legendre_integrate_1d(integrand_1, 0, dx, n=n_gl)
            i2 = gauss_legendre_integrate_1d(integrand_2, 0, dx, n=n_gl)
            if(i==0):
                B[idx] += i2
            elif(i==Nx-1):
                B[idx] += i1
            else:
                B[idx] = i1+i2
    # Top boundary 
    if(qt is not None):
        for i in range(Nx):
            idx = i + (Ny-1)*Nx
            x_i = xg[idx]
            integrand_1 = lambda s: qt(x_i + dx - s)*(s/dx)
            integrand_2 = lambda s: qt(x_i - s)*(1-s/dx)
            i1 = gauss_legendre_integrate_1d(integrand_1, 0, dx, n=n_gl)
            i2 = gauss_legendre_integrate_1d(integrand_2, 0, dx, n=n_gl)
            if(i==0):
                B[idx] += i1
            elif(i==Nx-1):
                B[idx] += i2
            else:
                B[idx] = i1+i2

    return B



def apply_dirichlet_bcs(A, b, Nx, Ny, xg, yg, u):
    [ut, ub, ul, ur] = u
    for i in range(Nx): # Bottom and top rows
        if(ub is not None):
            idx = i # Node index for bottom row
            x_i = xg[idx]
            A[idx,:]=0.0
            A[idx,idx]=1.0
            b[idx] = ub(x_i)
        if(ut is not None):
            idx = i+(Ny-1)*Nx # Node index for top row
            x_i = xg[idx]
            A[idx,:]=0.0
            A[idx,idx]=1.0
            b[idx] = ut(x_i)

    for j in range(Ny):
        if(ul is not None):
            idx = j * Nx
            y_i = yg[idx]
            A[idx, :] = 0.0
            A[idx, idx] = 1.0
            b[idx] = ul(y_i)
        if(ur is not None):
            idx = j * Nx + (Nx - 1)
            y_i = yg[idx]
            A[idx, :] = 0.0
            A[idx, idx] = 1.0
            b[idx] = ur(y_i)

    return A, b


if __name__ == '__main__':
    create_mass_stiffness_matrices(1,1,1)