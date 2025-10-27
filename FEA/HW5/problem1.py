import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

if __name__ == '__main__':
    h=10
    EI = 1e9 * 1e-3
    x = sp.Symbol('x')
    phis = [
        1-3*(x/h)**2 + 2*(x/h)**3,
        -x*(1-x/h)**2,
        3*(x/h)**2 -2*(x/h)**3,
        -x*((x/h)**2-x/h)
    ]
    dphis = [sp.diff(p,x,2) for p in phis]

    k_e = np.zeros((4,4))
    for i in range(4):
        for j in range(4):
            k_e[i,j] = sp.integrate(dphis[i]*dphis[j],(x,0,10))

    print(k_e)
    f_e1 = np.zeros(4)
    for i in range(4):
        f_e1[i] = sp.integrate(phis[i]*100,(x,0,10))
    print(f_e1)
    f_e2 = np.zeros(4)
    for i in range(4):
        f_e2[i] = 1000*phis[0].subs(x,5).evalf()
    print(f_e2)

    k_g = np.zeros((6,6))
    k_g[0:4,0:4] += k_e
    k_g[2:6,2:6] += k_e
    k_g *= EI
    print(k_g)

    f_g = np.zeros((6))
    f_g[0:4] += f_e1
    f_g[2:6] += f_e2
    print(f_g)

    Q_g = np.array([0,0,0,500,0,0])
    F = f_g + Q_g

    # Modify 1st and 5th DOF for dirichlet BC enforcement
    k_g_old = np.copy(k_g)
    k_g[0,:]=0.0
    k_g[4,:]=0.0
    k_g[0,0]=1
    k_g[4,4]=1
    F[0]=0
    F[4]=0

    u=np.linalg.solve(k_g,F)
    print(u)

    Q = k_g_old@u - f_g
    print(Q)

    # --- Plotting code for parts d and e ---

    # Define shape functions as lambdas for numerical evaluation
    h = 10
    phis_num = [
        sp.lambdify('x', 1-3*(x/h)**2 + 2*(x/h)**3, 'numpy'),
        sp.lambdify('x', -x*(1-x/h)**2, 'numpy'),
        sp.lambdify('x', 3*(x/h)**2 -2*(x/h)**3, 'numpy'),
        sp.lambdify('x', -x*((x/h)**2-x/h), 'numpy')
    ]

    # Part d: Plot transverse displacement for each element
    x_elem = np.linspace(0, h, 100)
    # Element 1: nodes 0-3 (u[0:4])
    u_elem1 = np.zeros_like(x_elem)
    for i in range(4):
        u_elem1 += u[i] * phis_num[i](x_elem)
    plt.figure()
    plt.plot(x_elem, u_elem1)
    plt.title('Element 1 Transverse Displacement')
    plt.xlabel('x [Element 1] (m)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)

    # Element 2: nodes 2-5 (u[2:6])
    u_elem2 = np.zeros_like(x_elem)
    for i in range(4):
        u_elem2 += u[i+2] * phis_num[i](x_elem)
    plt.figure()
    plt.plot(x_elem, u_elem2)
    plt.title('Element 2 Transverse Displacement')
    plt.xlabel('x [Element 2] (m)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)

    # Part e: Plot transverse displacement over the entire beam
    x_full = np.linspace(0, 2*h, 200)
    u_full = np.zeros_like(x_full)
    for idx, xval in enumerate(x_full):
        if xval <= h:
            # Element 1
            xi = xval
            u_full[idx] = sum(u[i] * phis_num[i](xi) for i in range(4))
        else:
            # Element 2
            xi = xval - h
            u_full[idx] = sum(u[i+2] * phis_num[i](xi) for i in range(4))
    plt.figure()
    plt.plot(x_full, u_full)
    plt.title('Transverse Displacement Over Entire Beam')
    plt.xlabel('x (m)')
    plt.ylabel('Displacement (m)')
    plt.grid(True)

    plt.show()


