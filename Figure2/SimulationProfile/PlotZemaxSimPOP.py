import numpy as np, matplotlib.pyplot as plt
import codecs


data = np.loadtxt("zemaxprofile.txt")
x, y = data[:, 0], data[:, 1]
maxVal = np.max(y)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(x*1000, y/maxVal, lw=1.6, c=(0, 176/255, 80/255, 1))
plt.ylabel("Intensity (norm.)", size=15)
plt.xlabel(r"x ($\mu$m)", size=15)
plt.tick_params(labelsize=15)
#plt.vlines(x=4, ymin=1e-9, ymax=1, ls='--', color='k', lw=1.5)
#plt.vlines(x=-4, ymin=1e-9, ymax=1, ls='--', color='k', lw=1.5)
plt.yscale('log')
plt.grid(True, which='both')
ax.set_xticks(np.arange(-8, 9, 4))
plt.xlim(-10, 10)
plt.ylim(5e-8, 1)
plt.show()
