import numpy as np
from matrices_2d import create_mass_stiffness_matrices, assemble_global_force_vector, assemble_global_mass_stiffness_matrices, \
apply_dirichlet_bcs, calc_all_fe
from geometry_2d import create_xy_global, create_element_mappings

def solve_2d_heat(Nx, Ny, dx, dy, alpha, Nt, dt, u0, u_bound, f):
    xg, yg = create_xy_global(Nx, Ny, dx, dy)
    element_map = create_element_mappings(Nx, Ny)

    Me, Ke = create_mass_stiffness_matrices(alpha, dx, dy)
    Mg, Kg = assemble_global_mass_stiffness_matrices(Me, Ke, element_map, Nx*Ny)

    U = np.zeros((Nt+1, Nx*Ny))
    U[0,:] = u0

    A = Mg/dt - Kg
    for timestep in range(Nt):
        if(timestep%10==0):
            print(f"Time step {timestep}/{Nt} and max temp {np.max(U[timestep,:])}")
        t = timestep * dt
        # Evaluate forcing function at current time step
        fe_arr = calc_all_fe(lambda x,y: f(x,y,t+dt), element_map, xg, yg) if f is not None else 0
        # fg = assemble_global_force_vector(fe_arr, element_map, Nx*Ny)
        RHS = 1/dt * Mg @ U[timestep,:] 
        A_mod, b_mod = apply_dirichlet_bcs(A.copy(), RHS.copy(), Nx, Ny, xg, yg, u_bound)
        U[timestep+1,:] = np.linalg.solve(A_mod, b_mod)

    return U, xg, yg, element_map

if __name__ == '__main__':
    from plotting_2d import plot_solution

    Nx, Ny = 10, 20
    dx, dy = 1.0, 1.0
    alpha = 1
    Nt = 1000
    dt = 0.1
    u0 = np.ones(Nx*Ny)
    u_bound = [lambda s: 0] * 4 # Homogenous Dirichlet BCs
    f = lambda x, y, t: np.zeros_like(x) # No forcing by default

    U, xg, yg, element_map = solve_2d_heat(Nx, Ny, dx, dy, alpha, Nt, dt, u0, u_bound, None)

    # plot_solution(U[0,:], xg, yg, element_map, title='Initial Condition')
    for timestep in range(Nt):
        t = timestep * dt
        if timestep % 10 == 0:
            print(f'Calculating step {timestep}')
            print(f'Max temperature: {np.max(U[timestep,:])}')
            # plot_solution(U[timestep], xg, yg, element_map, title=f'Solution at t={t}')
    # plot_solution(U[-1,:], xg, yg, element_map, title=f'Solution at t={Nt*dt:.2f}')
