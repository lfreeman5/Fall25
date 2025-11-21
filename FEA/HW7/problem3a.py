import numpy as np

pts=np.array([[1,2],
              [4,1],
              [4.5,6]])
alpha = np.zeros(3)
beta = np.zeros(3)
gamma = np.zeros(3)
for i in range(3):
    j=np.mod(i+1,3)
    k=np.mod(i+2,3)
    print(f'i j k {i} {j} {k}')
    alpha[i] = pts[j,0]*pts[k,1]-pts[k,0]*pts[j,1]
    beta[i] = pts[j,1]-pts[k,1]
    gamma[i] = pts[k,0]-pts[j,0]


A = np.sum(alpha)/2
cent = np.sum(pts, axis=0)/3

S00, S11, S12, S21, S22 = np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3)), np.zeros((3,3))
I20 = A/12 * (9*cent[0]**2+np.sum(pts[:,0]**2))
I02 = A/12 * (9*cent[1]**2+np.sum(pts[:,1]**2))
I11 = A/12 * (9*cent[0]*cent[1] + np.sum(pts[:,0]*pts[:,1]))
print(I20)
print(I02)
print(I11)

for i in range(3):
    for j in range(3):
        ai, aj = alpha[i], alpha[j]
        bi, bj = beta[i], beta[j]
        gi, gj = gamma[i], gamma[j]
        S00[i,j] = 1/(4*A)*(ai*aj+cent[0]*(ai*bj+aj*bi)+cent[1]*(ai*gj+aj*gi)+
                            1/A*(I20*bi*bj+I11*(gi*bj+gj*bi)+I02*gi*gj))
        S11[i,j] = 1/(4*A)*bi*bj
        S12[i,j] = 1/(4*A)*bi*gj
        S21[i,j] = 1/(4*A)*bj*gi
        S22[i,j] = 1/(4*A)*gi*gj

def print_matrix(mat, name):
    print(f"{name}:")
    for row in mat:
        print(" ".join(f"{val:10.4f}" for val in row))
    print()

print_matrix(S00, "S00")
print_matrix(S11, "S11")
print_matrix(S12, "S12")
print_matrix(S21, "S21")
print_matrix(S22, "S22")



print(10*A/3)

print_matrix(S00+S11+S12+S21+S22, 'S_all')