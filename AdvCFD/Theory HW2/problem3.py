import numpy as np

U_L = 1
U_R = 2
dx = 0.1
N = 100

def first_order_step(u):
    u_full = np.pad(u, (1,1), constant_values=(U_L,U_R))
    
    
