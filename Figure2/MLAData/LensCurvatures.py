import numpy as np, matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit as fit


#Load in the x and y profiles obtained from the profilometer and plot them
xFiles = glob.glob("MLA_Profiles/Row*_X.csv")
yFiles = glob.glob("MLA_Profiles/Row*_Y.csv")

xFiles = sorted(xFiles, key=lambda x: int(x.split('/')[1].split('_')[0][3:]))
yFiles = sorted(yFiles, key=lambda x: int(x.split('/')[1].split('_')[0][3:]))

#Load in x cross-section data
xData = []
for file in xFiles:
    xPos = []
    height = []
    with open(file, 'r') as data:
        for line in data.readlines()[4:]:
            x, y,_,_ = line.split(',')
            if y == 'NaN':
                continue
            
            xPos.append(x)
            height.append(y)
    tempArr = np.zeros((len(xPos), 2))
    tempArr[:, 0] = xPos
    tempArr[:, 1] = height
    maxIndex = np.argmax(tempArr[:, 1])
    tempArr[:, 0] -= tempArr[maxIndex, 0]
    tempArr[:, 1] -= tempArr[maxIndex, 1]
    xData.append(tempArr)

#Plot x cross-sections
for data in xData:
    
    x = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 0]
    y = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 1]/1000
    plt.plot(x,y, lw=2)

plt.tick_params(labelsize=18)
plt.xlabel("Distance ($\mu$m)", size=18)
plt.ylabel("Surface Height ($\mu$m)", size=18)
plt.show()

#Load in y cross-sections
yData = []
for file in yFiles:
    yPos = []
    height = []
    with open(file, 'r') as data:
        for line in data.readlines()[4:]:
            x, y,_,_ = line.split(',')
            if y == 'NaN':
                continue
            
            yPos.append(x)
            height.append(y)
    tempArr = np.zeros((len(yPos), 2))
    tempArr[:, 0] = yPos
    tempArr[:, 1] = height
    maxIndex = np.argmax(tempArr[:, 1])
    tempArr[:, 0] -= tempArr[maxIndex, 0]
    tempArr[:, 1] -= tempArr[maxIndex, 1]
    yData.append(tempArr)

#plot y cross-sections of the surface profile
for data in yData:
    
    x = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 0]
    y = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 1]/1000
    
    plt.plot(x,y)
plt.tick_params(labelsize=15)
plt.xlabel("Distance (um)", size=15)
plt.ylabel("Surface Height (um)", size=15)
plt.show()

#asphere Sag
def evenAsphere(r, c, k, a, r0):
    a0 = c*(r-r0)**2/(1 + np.sqrt(1-(1+k)*(c**2)*((r-r0)**2)))
    return a0 + a[0] * r**2 + a[1] * r**4 + a[2] * r**6


#---------------------------------
#Even asphere parameters from Zemax
AsphereParams = {}
asphereData = np.loadtxt("Asphere_Zemax_Design_Data.txt")
dictKeys = [str(0.525 + 0.025*i)[0:5] for i in range(0, 20)]
for i, key in enumerate(dictKeys):
    c, k = asphereData[i, 0:2]
    a = asphereData[i, 2:]
    AsphereParams[key] = (c, k, a)

#-----------------------------
#Plot the measured surface profile along with the one calculated from zemax
fig, ax = plt.subplots(4,5, sharex=True, sharey=True)
i, j = 0, 0
for data, EFL in zip(xData, AsphereParams.keys()):
    
    fitx = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 0]
    fity = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 1]/1000
    r = np.linspace(-50/1000,50/1000, len(fitx))
    
    ax[j, i].plot(fitx,fity)

    c, k, a = AsphereParams[EFL]
    ax[j, i].plot(r*1000, 1000*evenAsphere(r, c, k, a, 0))

    ax[j, i].tick_params(labelsize=15)
    ax[j, i].text(-20, 4, "{} um".format(float(EFL) * 1000))
   
    i += 1
    if i % 5 == 0:
        i = 0
        j += 1


fig.text(0.5, 0.04, 'Distance (um)', ha='center', size=15)
fig.text(0.04, 0.5, 'Surface Height (um)', va='center', rotation='vertical', size=15)
plt.show()
    

#--------------------------
#Plot deviations from the ideal aspheres
fig, ax = plt.subplots(2, 1, sharex=True, sharey=True)
for data, EFL in zip(xData, AsphereParams.keys()):
    
    fitx = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 0]
    fity = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 1]/1000
    c, k, a = AsphereParams[EFL]
    ax[0].plot(r*1000, 1000*evenAsphere(r, c, k, a, 0)-fity)



for data, EFL in zip(yData, AsphereParams.keys()):
    
    fitx = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 0]
    fity = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 1]/1000
    c, k, a = AsphereParams[EFL]
    ax[1].plot(r*1000, 1000*evenAsphere(r, c, k, a, 0)-fity)

fig.text(0.5, 0.01, 'Distance (um)', ha='center', size=15)
fig.text(0.01, 0.5, 'Error (um)', va='center', rotation='vertical', size=15)
plt.show()

#-------------------------------------------------------
#exctract curvatures from experimental data by using it as a fitting parameter
xRadExp = []
for data in xData:
    
    fitx = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 0]
    fity = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 1]/1000
    
    k = -1.002090
    a = np.array([1.176426, -3.855963, -15.820927])
    f = lambda r, R, r0: evenAsphere(r, 1/R, k, a, r0)
    params, cov = fit(f, fitx/1000, fity/1000, p0=[-0.2, 0])
    # plt.plot(fitx,fity)
    # plt.plot(fitx, f(fitx / 1000, *params)*1000)
    # plt.tick_params(labelsize=15)
    # plt.xlabel("Distance (um)", size=15)
    # plt.ylabel("Surface Height (um)", size=15)
    # plt.show()
    xRadExp.append(params[0])
    
xRadExp = np.array(xRadExp)
xRadDesign = np.array([AsphereParams[EFL][0] for EFL in AsphereParams.keys()])

#---------------------------------------------------------
#plot of errors in the radius of curvatures
err = (np.abs(1/xRadDesign) - np.abs(xRadExp)) * np.abs(xRadDesign) * 100
plt.plot(np.arange(1, 21), err, 'o')
plt.hlines(y = 0, xmin=0.0, xmax=20, ls='--', color='r')
plt.hlines(y = np.average(err), xmin=0.0, xmax=20, ls='--', color='g')
plt.xlabel("MLA Row", size=18)
plt.ylabel("(|R$_{design}$| - |R$_{exp}$|)/|R$_{design}$|  (%)", size=18)
plt.tick_params(labelsize=18)
plt.show()


#----------------------------------------------------------
#Sample plot for paper (lens 3):
EFL = list(AsphereParams.keys())[2]
data = xData[2]
    
#measured profile    
x = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 0]
y = data[np.logical_and(data[:,0] > -50, data[:, 0] <50), 1]/1000
plt.plot(x,y, lw=1.8, c='k')

#design profiles
c, k, a = AsphereParams[EFL]
plt.plot(r*1000, 1000*evenAsphere(r, c, k, a, 0), ls='--', c='r', lw=1.5)

plt.tick_params(labelsize=15)
plt.legend(["Measured", "Designed"], fontsize=12)
plt.xlabel(r"Distance ($\mu$m)", size=15)
plt.ylabel(r"Surface Height ($\mu$m)", size=15)
plt.show()

