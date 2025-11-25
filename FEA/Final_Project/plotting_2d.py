import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import animation

def plot_solution(U, xg, yg, element_map, title='FEA Solution'):
    """
    Plots the finite element solution U as a scalar color field by drawing each element.

    Args:
        U (np.ndarray): Solution vector of size (N_nodes).
        xg (np.ndarray): Global x-coordinates of all nodes.
        yg (np.ndarray): Global y-coordinates of all nodes.
        element_map (np.ndarray): Map of element nodes, shape (Ne, 4).
        title (str): Plot title.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert each quad to two triangles for triangulation
    triangles = []
    for element_nodes in element_map.astype(int):
        # [0, 1, 2, 3] assumed as [bl, br, tr, tl]
        n0, n1, n2, n3 = element_nodes
        # Triangle 1: [n0, n1, n2], Triangle 2: [n0, n2, n3]
        triangles.append([n0, n1, n2])
        triangles.append([n0, n2, n3])
    triangles = np.array(triangles)

    triang = Triangulation(xg, yg, triangles)
    tpc = ax.tripcolor(triang, U, shading='gouraud', cmap='jet')
    fig.colorbar(tpc, ax=ax, label='Solution Value')

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xg.min(), xg.max())
    ax.set_ylim(yg.min(), yg.max())

    plt.show()

def plot_comparison(U, xg, yg, element_map, u_exact_func, t_final, Lx=None, Ly=None, Nx_fine=100, Ny_fine=100):
    """
    Plots side-by-side color plots of computed U and exact solution evaluated at higher resolution.
    Both plots share the same color scale.
    Args:
        U (np.ndarray): Computed solution at nodes (N_nodes,).
        xg (np.ndarray): Node x-coordinates.
        yg (np.ndarray): Node y-coordinates.
        element_map (np.ndarray): Element connectivity.
        u_exact_func (callable): Function (x, y, t) -> u_exact.
        t_final (float): Time at which to evaluate the exact solution.
        Lx, Ly (float): Domain size. If None, inferred from xg, yg.
        Nx_fine, Ny_fine (int): Resolution for exact solution plot.
        title (str): Plot title.
    """
    if Lx is None:
        Lx = xg.max()
    if Ly is None:
        Ly = yg.max()

    title=f'Finite Element vs Exact Solution to Heat Eq. at t={t_final}'

    # Compute exact solution on a fine grid
    x_fine = np.linspace(0, Lx, Nx_fine)
    y_fine = np.linspace(0, Ly, Ny_fine)
    X_fine, Y_fine = np.meshgrid(x_fine, y_fine)
    U_exact_fine = u_exact_func(X_fine, Y_fine, t_final)

    # Get color scale limits
    vmin = min(U.min(), U_exact_fine.min())
    vmax = max(U.max(), U_exact_fine.max())

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    # Computed solution
    triangles = []
    for element_nodes in element_map.astype(int):
        n0, n1, n2, n3 = element_nodes
        triangles.append([n0, n1, n2])
        triangles.append([n0, n2, n3])
    triangles = np.array(triangles)
    triang = Triangulation(xg, yg, triangles)
    tpc = axes[0].tripcolor(triang, U, shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
    axes[0].set_title('Computed Solution')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_aspect('equal', adjustable='box')
    axes[0].set_xlim(0, Lx)
    axes[0].set_ylim(0, Ly)

    # Exact solution
    im = axes[1].imshow(U_exact_fine, extent=[0, Lx, 0, Ly], origin='lower', cmap='jet', vmin=vmin, vmax=vmax, aspect='equal')
    axes[1].set_title('Exact Solution')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].grid(False)  # Turn off grid for exact solution plot

    # Shared colorbar (use separate axis to avoid overlap)
    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(tpc, cax=cax, label='Solution Value')

    fig.suptitle(title)
    plt.show()

def animate_solution(U, xg, yg, element_map, dt, filename="solution_animation.mp4", interval=50, plot_every=1):
    """
    Animate the temporal solution and save to an .mp4 file.
    Args:
        U: (Nt+1, N_nodes) array of solutions at each timestep
        xg, yg: node coordinates
        element_map: element connectivity
        dt: timestep size
        filename: output mp4 filename
        interval: ms between frames
        plot_every: plot every Nth frame
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Convert each quad to two triangles for triangulation
    triangles = []
    for element_nodes in element_map.astype(int):
        n0, n1, n2, n3 = element_nodes
        triangles.append([n0, n1, n2])
        triangles.append([n0, n2, n3])
    triangles = np.array(triangles)
    triang = Triangulation(xg, yg, triangles)

    # Determine colorbar bounds from all time steps
    vmin = np.min(U)
    vmax = np.max(U)

    tpc = ax.tripcolor(triang, U[0], shading='gouraud', cmap='jet', vmin=vmin, vmax=vmax)
    cbar = fig.colorbar(tpc, ax=ax, label='Solution Value')
    ax.set_title('FEA Solution')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(xg.min(), xg.max())
    ax.set_ylim(yg.min(), yg.max())

    def update(frame):
        idx = frame * plot_every
        tpc.set_array(U[idx])
        ax.set_title(f"FEA Solution at t={idx*dt:.3f}")
        return tpc,

    num_frames = (U.shape[0] - 1) // plot_every + 1
    anim = animation.FuncAnimation(
        fig, update, frames=num_frames, interval=interval, blit=False
    )
    anim.save(filename, writer='ffmpeg', fps=1000//interval)
    plt.close(fig)