import numpy as np
from matrices_2d import create_mass_stiffness_matrices, assemble_global_force_vector, assemble_global_mass_stiffness_matrices, \
apply_dirichlet_bcs
from geometry_2d import create_xy_global, create_element_mappings
from plotting_2d import plot_solution

if __name__ == '__main__':
    Nx, Ny = 50,50
    dx, dy = 1.0, 1.0
    alpha = 1

    xg, yg = create_xy_global(Nx, Ny, dx, dy)
    element_map = create_element_mappings(Nx, Ny)

    Me, Ke = create_mass_stiffness_matrices(alpha, dx, dy)
    Mg, Kg = assemble_global_mass_stiffness_matrices(Me, Ke, element_map, Nx*Ny)

    u0 = np.ones(Nx*Ny)
    u_bound = [lambda s: 0] * 4 # Homogenous Dirichlet BCs


    Nt = 1000
    dt = 0.001

    U = np.zeros((Nt+1, Nx*Ny))
    U[0,:] = u0

    plot_solution(U[0,:], xg, yg, element_map, title='Initial Condition')

    A = Mg/dt - Kg
    for timestep in range(Nt):
        t=timestep*dt
        if(timestep%10==0):
            print(f'Calculating step {timestep}')
            plot_solution(U[timestep], xg, yg, element_map, title=f'Solution at t={t}')
        RHS = 1/dt * Mg @ U[timestep,:]
        A_mod, b_mod = apply_dirichlet_bcs(A.copy(),RHS.copy(),Nx,Ny,xg,yg,u_bound)
        U[timestep+1,:] = np.linalg.solve(A_mod, b_mod)

    plot_solution(U[-1,:], xg, yg, element_map, title=f'Solution at t={Nt*dt:.2f}')
