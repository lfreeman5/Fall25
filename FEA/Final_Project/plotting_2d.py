import numpy as np
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

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