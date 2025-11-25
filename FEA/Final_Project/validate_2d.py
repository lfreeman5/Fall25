import sympy as sp
import numpy as np
from geometry_2d import create_xy_global
from solve_2d_heat import solve_2d_heat 
from plotting_2d import plot_comparison

if __name__ == "__main__":
    Lx = np.pi
    Ly = np.pi
    Nx = 50
    Ny = 50
    N = Nx*Ny
    dx = Lx/(Nx-1)
    dy = Ly/(Ny-1)
    alpha = 1/2.
    Nt = 500
    dt = 0.001


    # Set up exact solution
    x,y,t = sp.symbols('x y t')
    u_exact = sp.sin(x)*sp.sin(y)*sp.exp(-t)
    symbolic_f = sp.diff(u_exact, t, 1) - alpha*(sp.diff(u_exact, x, 2)+sp.diff(u_exact, y, 2))
    print('f: ', sp.simplify(symbolic_f))

    # Lambdify symbolic_f to create a callable function f(x, y, t)
    f = sp.lambdify((x, y, t), symbolic_f, modules='numpy')
    u_0_expr = u_exact.subs(t, 0)
    u_0 = sp.lambdify((x, y), u_0_expr, modules='numpy')

    # Initial condition value
    xg, yg = create_xy_global(Nx, Ny, dx, dy)
    u_0_vals = np.zeros(N)
    for n in range(N):
        u_0_vals[n] = u_0(xg[n],yg[n])


    # Boundary condition functions
    u_L_expr = u_exact.subs(x, 0)           # Left boundary (x=0)
    u_R_expr = u_exact.subs(x, Lx)          # Right boundary (x=Lx)
    u_B_expr = u_exact.subs(y, 0)           # Bottom boundary (y=0)
    u_T_expr = u_exact.subs(y, Ly)          # Top boundary (y=Ly)
    u_L = sp.lambdify((y, t), u_L_expr, modules='numpy')
    u_R = sp.lambdify((y, t), u_R_expr, modules='numpy')
    u_B = sp.lambdify((x, t), u_B_expr, modules='numpy')
    u_T = sp.lambdify((x, t), u_T_expr, modules='numpy')
    dirichlet_bcs = [u_T, u_B, u_L, u_R]

    # Solve
    params = {
        'Nx': Nx,
        'Ny': Ny,
        'dx': dx,
        'dy': dy,
        'alpha': alpha,
        'Nt': Nt,
        'dt': dt,
        'u0': u_0_vals,
        'u_bound': dirichlet_bcs, # Homogenous Dirichlet BCs
        'q_bound': [lambda s,t:0]*4,
        'f': f # Calculated forcing function
    }
    U, xg, yg, element_map = solve_2d_heat(params)

    # Validate: compute norm of error at final timestep
    t_final = (Nt-1) * dt
    U_exact_vals = u_exact_func = sp.lambdify((x, y, t), u_exact, modules='numpy')
    U_exact_final = U_exact_vals(xg, yg, t_final)
    error = np.linalg.norm(U[-1] - U_exact_final)
    print(f"L2 norm of error at final timestep: {error}")

    # Plot side-by-side comparison
    plot_comparison(U[-1], xg, yg, element_map, U_exact_vals, t_final, Lx=Lx, Ly=Ly)

