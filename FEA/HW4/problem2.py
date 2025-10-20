import numpy as np

if __name__ == '__main__':
    h=16
    ea1=1.5e7
    ea2=3.75e6
    ea3=3.75e6
    ke = np.array([[1,-1],[-1,1]])

    K_g = np.zeros((4,4))
    K_g[0:2,0:2] += ke * ea1
    K_g[1:3,1:3] += ke * ea2
    K_g[2:4,2:4] += ke * ea3
    Q = np.zeros(4)
    Q[1]=5e3
    Q[-1]=-2e3

    K_g[0,:] = 0.0
    K_g[0,0] = 1

    u = np.linalg.solve(K_g, Q)
    u_prime = np.array([(u[i+1]-u[i])/16 for i in range(3)])
    ea = np.array([ea1,ea2,ea3])

    print(K_g/1e6)


    print(u*1e4)
    print(u_prime*1e4)
    print(u_prime*ea*16)
    print(u_prime*ea*16/(np.array([8,6,4])))
