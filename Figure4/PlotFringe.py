import numpy as np, matplotlib.pyplot as plt
from PIL import Image

dataLow = Image.open("lowContrast.bmp")
dataHigh = Image.open("HighContrast.bmp")
dataLow = np.array(dataLow)
dataHigh = np.array(dataHigh)

fig, ax = plt.subplots(1, 2)
pl = ax[0].imshow(dataLow/ np.max(dataHigh), aspect=1.7)
ax[0].tick_params(labelsize=15)
ax[1].imshow(dataHigh/np.max(dataHigh), aspect=1.7)
fig.text(0.5, 0.04, 'x (px)', ha='center', size=17)
fig.text(0.04, 0.5, 'y (px)', va='center', rotation='vertical', size=17)
ax[1].tick_params(labelsize=15)
fig.tight_layout()
fig.colorbar(pl)
plt.show()

#taking cross-section
y = 2166
dataLowY = dataLow[y, :]
dataHighY = dataHigh[y, :]

fig, ax = plt.subplots(1, 2)
ax[0].plot(dataLowY)
ax[1].plot(dataHighY)
ax[0].tick_params(labelsize=15)
fig.text(0.5, 0.04, 'x (px)', ha='center', size=17)
fig.text(0.04, 0.5, 'Intensity (ADU)', va='center', rotation='vertical', size=17)
ax[1].tick_params(labelsize=15)
plt.show()