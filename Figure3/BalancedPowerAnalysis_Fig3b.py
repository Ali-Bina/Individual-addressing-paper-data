import numpy as np, matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

Pin = 489 #uW
#includes losses due to objective, waveguide, waveguide bond and VGAs
Pout = np.array([2.793, 3.501, 3.936, 3.614, 3.783, 4.7, 2.307, 2.156, 3.056, 3.385,
4.514, 2.756, 2.477, 2.674, 1.405, 1.091]) #uW

IL_AOM = 10**(-3 / 10)
IL_Delays = 10**(-(3.25)/10) # average loss
Pout *= IL_AOM*IL_Delays

#Reducing the power of all AOM channels to that of the minimum
SortedPs = np.array(sorted(Pout))
Pout = np.array([SortedPs[i] * (16 - i) for i in range(16)])



#assuming 2 W input power - plot the power out of each channel
Rpower = Pout/Pin
power = Rpower[::-1] * 2 * 1000/np.arange(1,17)

#error analysis:
dIL = 0.4
dP = 0.03
dPin = dP * Pin
dPout = dP * Pout

dP_IL_AOM = power * np.log(10)/10 * dIL
dp_IL_Delay = power * np.log(10)/10 * dIL
dP_dPin = power/Pin * dPin
dP_dPout = power / Pout * dPout

eTot = np.sqrt(dP_IL_AOM**2 + dp_IL_Delay**2 + dP_dPin**2 + dP_dPout**2)


#plot data
plt.errorbar(np.arange(1, 17), power,eTot, fmt='o', c='k', lw=1.5, capsize=2)
plt.xlabel("Number of Channels", size=15)
plt.ylabel("Power (mW)", size=15)
plt.tick_params(labelsize=15)
plt.xlabel("Number of Channels", size=15)
plt.xlim(0.85, 16.15)
ax = plt.gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.show()
