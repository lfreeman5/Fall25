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

def eval_f_e(f,xi,yi,xf,yf):
    '''
    f = f(x,y), callable
    '''
    a = xf - xi
    b = yf - yi
    phis = [lambda x, y: (1 - x/a)*(1 - y/b),
            lambda x, y: (x/a)*(1 - y/b),
            lambda x, y: (x/a)*(y/b),
            lambda x, y: (1 - x/a)*(y/b)]
    f_e = np.zeros(4)
    for i in range(4):
        # Integrate phi[i](x, y) * f(xi + x, yi + y) over x in [0, a], y in [0, b]
        integrand = lambda y, x: phis[i](x, y) * f(xi + x, yi + y)
        f_e[i], _ = dblquad(integrand, 0, a, lambda x: 0, lambda x: b)
    return f_e

def calc_all_fe(f, element_map, xg, yg):
    Ne = element_map.shape[0]
    fe_arr = np.zeros((Ne,4))
    for i,e in enumerate(element_map):
        xi, yi = xg[e[0]], yg[e[0]]
        xf, yf = xg[e[2]], yg[e[2]]
        fe_arr[i] = eval_f_e(f,xi,yi,xf,yf)

    return fe_arr

def calc_boundary_term(Nx, Ny, dx, dy, xg, yg, q):
    '''
    xg, yg are N=Nx*Ny arrays of global position
    q is the arrays of boundary vectors
    '''
    [qt,qb,ql,qr] = q
    N = Nx*Ny
    B = np.zeros(N)
    phix1 = lambda s: 1+s/dx
    phix2 = lambda s: 1-s/dx
    phiy1 = lambda s: 1+s/dy
    phiy2 = lambda s: 1-s/dy
    for i in range(Nx): # Bottom and top rows
        idx = i # Node index for bottom row
        x_i = xg[idx]
        q = lambda s: qb(s+x_i)
        integrand_1 = lambda s: q(s)*phix1(s)
        integrand_2 = lambda s: q(s)*phix2(s)
        _, i1 = quad(integrand_1, -dx, 0)
        _, i2 = quad(integrand_2, 0, dx)
        if(i==0): # For LHS nodes, only include right integral
            B[idx] = i2
        elif(i==Nx-1): # For RHS nodes, only include left integral
            B[idx] = i1
        else: 
            B[idx] = i1+i2


        idx = i+(Ny-1)*Nx # Node index for top row
        x_i = xg[idx]
        q = lambda s: qt(s+x_i)
        integrand_1 = lambda s: q(s)*phix1(s)
        integrand_2 = lambda s: q(s)*phix2(s)
        _, i1 = quad(integrand_1, -dx, 0)
        _, i2 = quad(integrand_2, 0, dx)
        if(i==0): # For LHS nodes, only include right integral
            B[idx] = i2
        elif(i==Nx-1): # For RHS nodes, only include left integral
            B[idx] = i1
        else: 
            B[idx] = i1+i2

    for j in range(Ny):
        idx = j*Nx # Node index for LHS
        y_i = yg[idx]
        q = lambda s: ql(s+y_i)
        integrand_1 = lambda s: q(s)*phiy1(s)
        integrand_2 = lambda s: q(s)*phiy2(s)
        _, i1 = quad(integrand_1, -dy, 0)
        _, i2 = quad(integrand_2, 0, dy)
        if(j==0): # For bottom nodes, only include up integral
            B[idx] = i2
        elif(j==Ny-1): # For top nodes, only include down integral
            B[idx] = i1
        else: 
            B[idx] = i1+i2

        idx = j*Nx + (Nx - 1) # Node index for LHS
        y_i = yg[idx]
        q = lambda s: qr(s+y_i)
        integrand_1 = lambda s: q(s)*phiy1(s)
        integrand_2 = lambda s: q(s)*phiy2(s)
        _, i1 = quad(integrand_1, -dy, 0)
        _, i2 = quad(integrand_2, 0, dy)
        if(j==0): # For bottom nodes, only include up integral
            B[idx] = i2
        elif(j==Ny-1): # For top nodes, only include down integral
            B[idx] = i1
        else: 
            B[idx] = i1+i2


def apply_dirichlet_bcs(A, b, Nx, Ny, xg, yg, u):
    [ut, ub, ul, ur] = u
    for i in range(Nx): # Bottom and top rows
        idx = i # Node index for bottom row
        x_i = xg[idx]
        A[idx,:]=0.0
        A[idx,idx]=1.0
        b[idx] = ub(x_i)

        idx = i+(Ny-1)*Nx # Node index for top row
        x_i = xg[idx]
        A[idx,:]=0.0
        A[idx,idx]=1.0
        b[idx] = ut(x_i)

    for j in range(Ny):
        idx = j * Nx
        y_i = yg[idx]
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        b[idx] = ul(y_i)

        idx = j * Nx + (Nx - 1)
        y_i = yg[idx]
        A[idx, :] = 0.0
        A[idx, idx] = 1.0
        b[idx] = ur(y_i)

    return A, b


if __name__ == '__main__':
    create_mass_stiffness_matrices(1,1,1)