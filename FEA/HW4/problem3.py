import numpy as np
pi=np.pi

if __name__ == '__main__':
    h=16
    ea1=pi*4**2/4 * 30e6 / 12
    ea2=pi*2.5**2/4 * 10e6 / 8
    ea3=pi*2**2/4 * 30e6 / 10
    ke = np.array([[1,-1],[-1,1]])

    K_g = np.zeros((4,4))
    K_g[0:2,0:2] += ke * ea1
    K_g[1:3,1:3] += ke * ea2
    K_g[2:4,2:4] += ke * ea3

    print(K_g/1e6/pi)

    Q = np.zeros(4)
    Q[2]=-20e3
    Q[-1]=5e3

    K_g[0,:] = 0.0
    K_g[0,0] = 1

    u = np.linalg.solve(K_g, Q)
    L = [12,8,10]
    A = np.array([4,2.5,2])**2/4*pi
    u_prime = np.array([(u[i+1]-u[i])/L[i] for i in range(3)])
    ea = np.array([ea1,ea2,ea3])

    print(u*1e4)
    print(u_prime*1e4)
    print(u_prime*ea*np.array(L))
    print(u_prime*ea*np.array(L)/A)
