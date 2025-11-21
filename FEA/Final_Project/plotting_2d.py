import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

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
    
    patches = []
    colors = []

    # The node ordering is now counter-clockwise:
    # [bottom-left, bottom-right, top-right, top-left]
    # This corresponds to indices [0, 1, 2, 3] from element_map rows.
    poly_node_indices = [0, 1, 2, 3]

    for i, element_nodes in enumerate(element_map.astype(int)):
        # Get coordinates for the polygon
        polygon_nodes = element_nodes[poly_node_indices]
        polygon_coords = np.array([xg[polygon_nodes], yg[polygon_nodes]]).T
        
        # Create a polygon patch
        polygon = Polygon(polygon_coords, closed=True)
        patches.append(polygon)
        
        # Color is based on the average value of the solution at the element's nodes
        avg_solution = np.mean(U[element_nodes])
        colors.append(avg_solution)

    p = PatchCollection(patches, cmap='jet', alpha=1.0)
    p.set_array(np.array(colors))
    ax.add_collection(p)
    fig.colorbar(p, ax=ax, label='Solution Value')

    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal', adjustable='box')
    
    # Set limits to see the whole mesh
    ax.set_xlim(xg.min(), xg.max())
    ax.set_ylim(yg.min(), yg.max())
    
    plt.show()