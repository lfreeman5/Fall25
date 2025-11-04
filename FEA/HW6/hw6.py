import numpy as np
E=207E9
A = 5/100 * 1/100 
h1=1

K_g = np.zeros((6,6))
K_g[0,:] = np.array([1.5,0.5,-1,0,-0.5,-0.5])
K_g[1,:] = np.array([0,0,0,0,-0.5,-0.5])
K_g[2,:] = np.array([-1,0,1,0,0,0])
K_g[3,:] = np.array([0,0,0,1,0,-1])
K_g[4,:] = np.array([-0.5,-0.5,0,0,0.5,0.5])
K_g[5,:] = np.array([-0.5,-0.5,0,-1,0.5,1.5])
K_g = K_g*E*A/h1

T_c = np.eye(6)
beta = np.deg2rad(60)
cb, sb = np.cos(beta), np.sin(beta)
T_c[2:4,2:4] = np.array([[cb,-sb],[sb,cb]])

Q_applied = np.array([0,0,0,1E6,0,0]).T
Q_applied_c = T_c.T@Q_applied

K_c = T_c.T@K_g@T_c
print(K_c[2,:])
u3c = Q_applied_c[2]/K_c[2,2]
print(u3c)
u_c = np.zeros(6)
u_c[2] = u3c
u = T_c@u_c
Q = K_g@u - Q_applied

print(u)
print(Q)
print(u[3]*E/1e6)

Q_e = E*A/h1 * np.array([[1,-1],[-1,1]]) @ np.array([u[3],u[5]]).T - np.array([1E6,0]).T
print(Q_e)
print(Q_e/A/1e6)