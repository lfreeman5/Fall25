import numpy as np

def create_xy_global(Nx, Ny, dx, dy):
    '''
    all Ns here refer to number of nodes
    '''
    N = Nx*Ny
    x,y=np.zeros(N), np.zeros(N)
    for i in range(Nx):
        for j in range(Ny):
            x[i+j*Nx] = i*dx
            y[i+j*Nx] = j*dy
    return x,y

def create_element_mappings(Nx, Ny):
    '''
    Nx and Ny are the number of nodes
    Number of elements in each direction is -1 of that
    '''
    N = (Nx-1)*(Ny-1)
    element_to_global_map = np.zeros((N,4))
    for i in range(Nx-1):
        for j in range(Ny-1):
            element_to_global_map[i+j*(Nx-1),0] = j*Nx + i
            element_to_global_map[i+j*(Nx-1),1] = j*Nx + i + 1
            element_to_global_map[i+j*(Nx-1),2] = (j+1)*Nx + i +1
            element_to_global_map[i+j*(Nx-1),3] = (j+1)*Nx + i 
    return element_to_global_map.astype(int)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # Parameters for the grid
    Nx, Ny = 7, 4  # number of nodes in x and y
    dx, dy = 1.0, 1.0

    # Generate node coordinates
    x, y = create_xy_global(Nx, Ny, dx, dy)

    # Generate element to global node mapping
    element_map = create_element_mappings(Nx, Ny)

    for i,e in enumerate(element_map):
        print(f'Element {i } nodes: {e}')

    # Plot nodes
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, color='red', zorder=5)
    for idx, (xi, yi) in enumerate(zip(x, y)):
        plt.text(xi+0.04, yi+0.04, f'{idx}', color='red', ha='left', fontsize=10)  # shifted right

    # Plot and label elements
    for e, nodes in enumerate(element_map.astype(int)):
        # Get the coordinates of the element's nodes in the correct order
        ex = x[nodes]
        ey = y[nodes]
        # Draw the element edges in the order 1-2, 2-3, 3-4, 4-1
        # The nodes array is [n1, n2, n3, n4] and should be connected as:
        # n1-n2, n2-n4, n4-n3, n3-n1 (counterclockwise)
        edge_pairs = [(0,1), (1,3), (3,2), (2,0)]
        for a, b in edge_pairs:
            plt.plot([ex[a], ex[b]], [ey[a], ey[b]], 'b-')
        # Compute centroid for label
        cx, cy = np.mean(ex), np.mean(ey)
        plt.text(cx, cy, f'E{e}', color='blue', ha='center', va='center', fontsize=10, bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
        # Print the node indices for this element inside the element
        plt.text(cx, cy-0.2, f'{nodes.tolist()}', color='black', ha='center', va='center', fontsize=8)

    plt.title('2D Quadrilateral Grid: Node and Element Labels')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.gca().set_aspect('equal')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.show()
