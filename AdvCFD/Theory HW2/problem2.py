import numpy as np
import matplotlib.pyplot as plt

n = 600
t = np.arange(1, n+1)

# Initial values
v = np.zeros(n)
w = np.zeros(n)
v[0] = 0.11110101010101
w[0] = 0.11110101010102

# Iterate Frisch map (recurrence)
for j in range(1, n):
    v[j] = 1 - .2 * v[j-1]**2 
    w[j] = 1 - .2 * w[j-1]**2 

# First plot: trajectories of v and w
plt.figure(figsize=(6, 6))
plt.plot(t, v, 'ro-', label='v (u)')
plt.plot(t, w, 'bo-', label='w (r)')
plt.xlabel('Iteration')
plt.ylabel('Value')
plt.title('Frisch Map Iterations')
plt.legend()
plt.xlim(1, n)
plt.ylim(-1, 1)
# No aspect ratio set, just square figure

# Second plot: error vs model
d0 = abs(v[0] - w[0])
model = d0 * np.exp2(t)  # Use exp2 for large t
model = np.minimum(2, model)

print(f'last error: {np.abs(w-v)[-1]}')

plt.figure(figsize=(6, 6))
plt.semilogy(t, np.abs(w - v), 'g.-', label='|w - v|')
plt.semilogy(t, model, 'k-', linewidth=1.2, label='Model')
plt.xlabel('Iteration')
plt.ylabel('Error (log scale)')
plt.title('Error Growth')
plt.legend()
plt.xlim(1, n)
# No aspect ratio set, just square figure

plt.show()