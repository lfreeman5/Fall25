import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------------------
# Parameter theta
# -----------------------------------------------
th = np.linspace(0, 2*np.pi, 1001)
i = 1j
bdfext3 = (11*np.exp(i*th) - 18 + 9*np.exp(-1*i*th) -2*np.exp(-2*i*th))/(6*(3-3*np.exp(-1*i*th)+np.exp(-2*i*th)))
plt.figure(figsize=(7,5))
plt.axis('equal')
plt.plot(bdfext3.real, bdfext3.imag, 'b-',label='BDF-EXT3')
plt.legend()
plt.ylabel('Im λΔt')
plt.xlabel('Re λΔt')
plt.show()
