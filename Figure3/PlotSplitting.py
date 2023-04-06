import numpy as np, matplotlib.pyplot as plt

splitting = np.loadtxt("TempData.csv", delimiter=',')
temp=["20", "22", "23", "26"]
Xlabels = [str(i) for i in range(1, 18)]
Xvals = np.array(range(17))

pTot = []
for i in range(4):
    pTot.append(np.sum(splitting[1:,i]))

for i in range(4):
    splitting[:, i] = splitting[:, i] / pTot[i] * 100

err=3/100
errors=[]
for i in range(4):
    errors.append(err * splitting[1:, i])

plt.bar(Xvals - 0.2, splitting[1:, 0], 0.2, yerr=errors[0],  label='T = ' + temp[0], capsize=2)
plt.bar(Xvals, splitting[1:, 1], 0.2, label='T = ' + temp[1], yerr=errors[1], capsize=2)
plt.bar(Xvals + 0.2, splitting[1:, 2], 0.2, label='T = ' + temp[2], yerr=errors[2], ecolor='black', capsize=2)
plt.bar(Xvals + 0.4, splitting[1:, 3], 0.2, label='T = ' + temp[3], yerr=errors[3], ecolor='black', capsize=2)
plt.xticks(Xvals, Xlabels)
plt.xlabel("Channels", size=15)
plt.ylabel("Power Output (%)", size=15)
plt.tick_params(labelsize=15)
plt.legend(fontsize=15)
plt.show()

n = splitting.shape[0]
maxChange = []
for i in range(1, n):
    maxChange.append(np.abs(splitting[i, 0] - splitting[i, 3]))