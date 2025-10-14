import numpy as np
import time

# original version
def create_lagrange_poly_original(i, x_arr):
    N = len(x_arr)-1
    denom = np.prod([x_arr[i]-x_arr[j] for j in range(N+1) if j!=i])
    def L_i(x):
        value = 1.0
        for j in range(N+1):
            if j == i: 
                continue
            value *= (x - x_arr[j]) 
        return value/denom
    return L_i

# optimized vectorized version
def create_lagrange_poly_fast(i, x_arr):
    N = len(x_arr)-1
    denom = np.prod([x_arr[i]-x_arr[j] for j in range(N+1) if j != i])
    others = [x_arr[j] for j in range(N+1) if j != i]
    def L_i(x):
        x = np.asarray(x)
        return np.prod([(x - xj) for xj in others], axis=0) / denom
    return L_i

# setup
N = 100
x_arr = np.linspace(-1, 1, N+1)
x_eval = np.linspace(-1, 1, 10000)
i = 5

L_orig = create_lagrange_poly_original(i, x_arr)
L_fast = create_lagrange_poly_fast(i, x_arr)

# time original
t0 = time.perf_counter()
y1 = np.array([L_orig(x) for x in x_eval])
t1 = time.perf_counter() - t0

# time fast
t2 = time.perf_counter()
y2 = L_fast(x_eval)
t3 = time.perf_counter() - t2

print(f"Original time: {t1:.5f} s")
print(f"Vectorized time: {t3:.5f} s")
print(f"Speedup factor: {t1/t3:.1f}x")
