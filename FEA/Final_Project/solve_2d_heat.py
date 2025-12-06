import numpy as np
from matrices_2d import create_mass_stiffness_matrices, assemble_global_force_vector, assemble_global_mass_stiffness_matrices, \
apply_dirichlet_bcs, calc_all_fe, calc_boundary_term
from geometry_2d import create_xy_global, create_element_mappings

import scipy.linalg

def solve_2d_heat(params):
    Nx = params['Nx']
    Ny = params['Ny']
    dx = params['dx']
    dy = params['dy']
    alpha = params['alpha']
    Nt = params['Nt']
    dt = params['dt']
    u0 = params['u0']
    u_bound = params['u_bound']
    q_bound = params['q_bound']
    f = params.get('f', None)

    xg, yg = create_xy_global(Nx, Ny, dx, dy)
    element_map = create_element_mappings(Nx, Ny)

    Me, Ke = create_mass_stiffness_matrices(alpha, dx, dy)
    Mg, Kg = assemble_global_mass_stiffness_matrices(Me, Ke, element_map, Nx*Ny)

    U = np.zeros((Nt+1, Nx*Ny))
    U[0,:] = u0

    [ut, ub, ul, ur] = u_bound
    [qt, qb, ql, qr] = q_bound

    A = Mg/dt - Kg

    # Prefactorize A_mod (with Dirichlet BCs applied for t=0)
    u_bound_timestep = [
        (lambda x, ut=ut: ut(x, 0+dt)) if ut is not None else None,
        (lambda x, ub=ub: ub(x, 0+dt)) if ub is not None else None,
        (lambda y, ul=ul: ul(y, 0+dt)) if ul is not None else None,
        (lambda y, ur=ur: ur(y, 0+dt)) if ur is not None else None
    ]
    A_mod, _ = apply_dirichlet_bcs(A.copy(), np.zeros(Nx*Ny), Nx, Ny, xg, yg, u_bound_timestep)
    lu, piv = scipy.linalg.lu_factor(A_mod)

    for timestep in range(Nt):
        t = timestep * dt
        if(timestep%25==0):
            print(f'Evaluating at t={t+dt} at timestep {timestep+1}/{Nt}')

        q_bound_timestep = [
            (lambda x, qt=qt: qt(x, t+dt)) if qt is not None else None,
            (lambda x, qb=qb: qb(x, t+dt)) if qb is not None else None,
            (lambda y, ql=ql: ql(y, t+dt)) if ql is not None else None,
            (lambda y, qr=qr: qr(y, t+dt)) if qr is not None else None
        ]
        B_term = calc_boundary_term(Nx, Ny, dx, dy, xg, yg, q_bound_timestep)

        fe_arr = None
        fg = 0
        if f is not None and getattr(f, "__name__", "") != "<lambda>":
            fe_arr = calc_all_fe(lambda x,y: f(x,y,t+dt), element_map, xg, yg)
            fg = assemble_global_force_vector(fe_arr, element_map, Nx*Ny)

        RHS = 1/dt * Mg @ U[timestep,:] + B_term
        if fe_arr is not None:
            RHS += fg

        u_bound_timestep = [
            (lambda x, ut=ut: ut(x, t+dt)) if ut is not None else None,
            (lambda x, ub=ub: ub(x, t+dt)) if ub is not None else None,
            (lambda y, ul=ul: ul(y, t+dt)) if ul is not None else None,
            (lambda y, ur=ur: ur(y, t+dt)) if ur is not None else None
        ]
        _, b_mod = apply_dirichlet_bcs(A_mod, RHS.copy(), Nx, Ny, xg, yg, u_bound_timestep)
        U[timestep+1,:] = scipy.linalg.lu_solve((lu, piv), b_mod)

    return U, xg, yg, element_map

if __name__ == '__main__':
    from plotting_2d import plot_solution, animate_solution

    # --- Simulation setup ---
    # Set grid size (number of nodes in x and y directions)
    Nx = 15
    Ny = 15

    # Boundary conditions:
    # u_bound: [top, bottom, left, right] (Dirichlet, function of (s, t) or None)
    # q_bound: [top, bottom, left, right] (Neumann, function of (s, t) or None)
    u_bound = [None, lambda s,t: 2, None, None]  # Example: bottom boundary set to 2
    q_bound = [lambda s,t: s, None, lambda s,t: 0., lambda s,t: 0.]  # Example: top Neumann varies with s

    # Domain discretization
    dx = 1/(Nx-1)
    dy = 1/(Ny-1)

    # Physical and numerical parameters
    alpha = 1         # Diffusivity
    Nt = 100          # Number of time steps
    dt = 0.1          # Time step size
    u0 = np.ones(Nx*Ny)  # Initial condition (uniform)

    # Source term (forcing function)
    f = lambda x, y, t: 0.0  # No internal heat source by default

    # Pack parameters into a dictionary
    params = {
        'Nx': Nx,
        'Ny': Ny,
        'dx': dx,
        'dy': dy,
        'alpha': alpha,
        'Nt': Nt,
        'dt': dt,
        'u0': u0,
        'u_bound': u_bound,
        'q_bound': q_bound,
        'f': f
    }

    # --- Run the solver ---
    U, xg, yg, element_map = solve_2d_heat(params)

    # --- Visualization ---
    # To plot the solution at specific time steps, uncomment below:
    # plot_solution(U[0,:], xg, yg, element_map, title='Initial Condition')
    for timestep in range(params['Nt']):
        t = timestep * params['dt']
        if timestep % 10 == 0:
            print(f'Plotting step {timestep}')
            print(f'Max temperature: {np.max(U[timestep,:])}')
            # plot_solution(U[timestep], xg, yg, element_map, title=f'Solution at t={t}')
    # plot_solution(U[-1,:], xg, yg, element_map, title=f'Solution at t={params["Nt"]*params["dt"]:.2f}')

    # --- Save animation of the temporal solution ---
    animate_solution(U, xg, yg, element_map, params['dt'], filename="solution_animation.mp4", interval=50, plot_every=10)
    print("Animation saved to solution_animation.mp4")
