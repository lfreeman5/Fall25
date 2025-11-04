import numpy as np

E=29E6
abe=19.5
acf=16.8
hbe=10*12
hcf=8*12

S = np.zeros((4,4))
S[0,:] = np.array([E*abe/hbe,0,1,0])
S[1,:] = np.array([0,E*acf/hcf,0,1])
S[2,:] = np.array([0,0,1,1])
S[3,:] = np.array([0,0,5,10])
F = np.zeros(4)
F[2] = 170000
F[3] = 1360000

u=np.linalg.solve(S,F)
print(u)
ub=u[0]
uc=u[1]
ua = (uc-ub)/5*(-5) +ub
ud = (uc-ub)/5*(12) +ub
print(ua)
print(ud)
