import numpy as np
import sympy as sp
from numpy import sin as s, cos as c

def T_e(beta):
    b=np.deg2rad(beta)
    return np.array([[c(b),s(b),0,0],[0,0,c(b),s(b)]])

def print_matrix(mat):
    mat = np.atleast_2d(mat)
    for row in mat:
        print(' '.join(f"{val:12.4f}" for val in row))
    print('\n')

ea1 = 20e6*2
ea2 = 20e6*2
ea3 = 10e6*2
h1,h2=10,10
h3 = np.sqrt(200)

k_e = np.array([[1,-1],[-1,1]])

b1 = -45
b2 = 225
b3 = -90
T1 = T_e(b1)
T2 = T_e(b2)
T3 = T_e(b3)

k_e1 = (ea1/h1)*T1.T@k_e@T1
print_matrix(k_e1*1e-6)
k_e2 = (ea2/h2)*T2.T@k_e@T2
print_matrix(k_e2*1e-6)
k_e3 = (ea3/h3)*T3.T@k_e@T3
print_matrix(k_e3*1e-6)

fe1 = np.ones(2)*0.5*10*h1
fe2 = np.ones(2)*0.5*10*h2
fe3 = np.zeros(2)
fe1 = T1.T@fe1
fe2 = T2.T@fe2
fe3 = T3.T@fe3
print_matrix(fe1)
print_matrix(fe2)

k_g = np.zeros((6,6))
k_g[0:4,0:4] += k_e1
k_g[2:6,2:6] += k_e2
k_g[0:2,0:2] += k_e3[0:2,0:2] # Top left corner
k_g[4:6,0:2] += k_e3[2:4,0:2] # Bottom left corner
k_g[0:2,4:6] += k_e3[0:2,2:4] # Top right corner
k_g[4:6,4:6] += k_e3[2:4,2:4] # Bottom right corner

print_matrix(k_g*1e-6)

f_g = np.zeros(6)
f_g[0:4] += fe1
f_g[2:6] += fe2
print_matrix(f_g)

Q_g = np.zeros(6)
Q_g[3]=-100

Tc = np.eye(6)
b=np.deg2rad(-45)
Tc[2:4,2:4] = np.array([[c(b),-s(b)],[s(b),c(b)]])
print_matrix(Tc)



fc = Tc.T@f_g
Qc = Tc.T@Q_g
K_c = Tc.T@k_g@Tc

K_c_mod = np.copy(K_c)
K_c_mod[0,:]*=0.
K_c_mod[:,0]*=0.
K_c_mod[0,0] = 1.
K_c_mod[2,:]*=0.
K_c_mod[:,2]*=0.
K_c_mod[2,2] = 1.
K_c_mod[3,:]*=0.
K_c_mod[:,3]*=0.
K_c_mod[3,3] = 1.
K_c_mod[4,:]*=0.
K_c_mod[:,4]*=0.
K_c_mod[4,4] = 1.
RHS_vec = fc+Qc
RHS_vec[0]=0.
RHS_vec[2]=0.
RHS_vec[3]=0.
RHS_vec[4]=0.

print(f'KCmod and RH mod')
print_matrix(K_c_mod)
print(RHS_vec)

u_c = np.linalg.solve(K_c_mod,RHS_vec)
print(u_c)

Q_c_solved = K_c@u_c - fc
print(Q_c_solved)

u = Tc @ u_c
u3 = np.array([u[0],u[1],u[4],u[5]])
u3L = T3 @ u3
Q3L = (k_e*ea3/h3)@u3L
print(Q3L)

u2 = np.array([u[2],u[3],u[4],u[5]])
u2L = T2 @ u2
Q2L = (k_e*ea2/h2)@u2L
print(Q2L)
print(u2L)
E2 = 20e6
sigma2 = u2L*E2/h2
print(sigma2)

print('We are checking \n\n')
k_g_mod = np.copy(k_g)
k_g_mod[0,:]*=0.
k_g_mod[:,0]*=0.
k_g_mod[0,0] = 1.
k_g_mod[2,:]*=0.
k_g_mod[:,2]*=0.
k_g_mod[2,2] = 1.
k_g_mod[3,:]*=0.
k_g_mod[:,3]*=0.
k_g_mod[3,3] = 1.
k_g_mod[4,:]*=0.
k_g_mod[:,4]*=0.
k_g_mod[4,4] = 1.
rhs_vec2 = Q_g+f_g
rhs_vec2[0]=0.
rhs_vec2[2]=0.
rhs_vec2[3]=0.
rhs_vec2[4]=0.
print(np.linalg.solve(k_g_mod,rhs_vec2))
print(k_g@np.linalg.solve(k_g_mod,rhs_vec2) - f_g)