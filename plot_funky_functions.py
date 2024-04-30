import numpy as np
import matplotlib.pyplot as plt

a = 1.07
b = 2.14
n = 6

x = np.linspace(-a*1.5, a*1.5, 100)
y = np.linspace(-b*1.5, b*1.5, 100)
X, Y = np.meshgrid(x, y)
z = np.tanh(((X/a)**n + (Y/b)**n -1)) + 1

fig, ax = plt.subplots()
cp = ax.contourf(X, Y, z)
ax.set_xlabel('X[m]')
ax.set_ylabel('Y[m]')
ax.set_title(r'$\tanh{((X/a)^n + (Y/b)^n -1)} + 1$')
clb = fig.colorbar(cp)
clb.ax.set_title('Cost')
plt.savefig("Graphs/plot_trampo_bed_in_peripheral_objective.png", dpi=300)
plt.show()
