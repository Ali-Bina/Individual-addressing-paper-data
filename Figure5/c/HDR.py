import numpy as np, matplotlib.pyplot as plt
import glob
from scipy.optimize import curve_fit as fit

#load file names of the profile data
files = glob.glob("*.txt")
files = sorted(files)

#extract exposure time from file name
tE = [x.split(".")[0] for x in files]

#Calibration Slopes calculated from signal vs power plots for each exposure time
calib = [99, 4052, 29068, 112981] #adu/uW

profiles = [np.loadtxt(file) for file in files]


#find maximum of non-saturated profile, use it to normalize all plots
data = profiles[0] / calib[0]
maxInd = np.unravel_index(np.argmax(data), data.shape)
xmax, ymax = maxInd
norm = np.max(data)
data /= norm

#number of pixels about the beam's peak to plot
n, m = 177, 177

pxSize = 1.12 #um


#Store the maximum intensity for each exposure and the corresponding index
maxes = []
for i in range(len(tE)):
    plot = profiles[i]
    Iy = plot[xmax-n:xmax+n, ymax] / calib[i] / norm
    maxes.append((np.max(Iy), np.argmax(Iy)))

#For each exposure, split the profile into a left and right half about the peak
LTot = []
RTot = []
for i in range(len(tE)):
    plot = profiles[i]
    Iy = plot[xmax-n:xmax+n, ymax] / calib[i] / norm
    ly = np.array(range(2*n)) * pxSize
    

    IyLeft = Iy[ly < ly[maxes[0][1]]]
    lyLeft = ly[ly < ly[maxes[0][1]]]
    LTot.append((IyLeft, lyLeft))

    IyRight = Iy[ly >= ly[maxes[0][1]]]
    lyRight = ly[ly >= ly[maxes[0][1]]]
    RTot.append((IyRight, lyRight))

#define sensor parameters for error analysis
E = 6.626e-34 * 2.998e8/(532e-9)
#couldn't find these values for the camera so I assumed the upperbounds given on
#https://www.microscopyu.com/tutorials/ccd-signal-to-noise-ratio#:~:text=Sliders%20are%20provided%20for%20varying,electrons%20per%20pixel%20per%20second).
DNR = 50 #electrons per pixel per second
RD = 20 #rms electrons per pixel
QE = 0.98 #https://raw.githubusercontent.com/khufkens/pi-camera-response-curves/master/Sony_IMX219.png


LT=[]
RT = []
ErrL=[]
ErrR=[]
#stitch the left halves at the different exposure together
#do the same for right halves
for i in range(len(tE)):

    #left half:

    IyLeft, lyLeft = LTot[i]
    #remove the saturated portion of the higher exposure images
    if i!=0:
        LI = IyLeft[IyLeft <= maxes[i][0]/1.01]
    else:
        LI = IyLeft

    #remove regions where consecutive exposures overlap 
    if i < len(tE) - 1:
        LI = LI[LI > maxes[i+1][0]]

    #error analysis
    numPhoton = LI*norm*float(tE[i])/1e12/E
    photonNoise = np.sqrt(numPhoton)
    darkNoise = DNR*float(tE[i])/1e6
    Err = np.sqrt(RD**2+photonNoise**2+darkNoise**2)

    LT= list(LI) + LT
    ErrL = list(Err) + ErrL

    #Repeat the same steps for the right half:

    IyRight, lyRight = RTot[i]
    if i != 0:
        RI = IyRight[IyRight <= maxes[i][0] / 1.01]
    else:
        RI = IyRight

    if i < len(tE) - 1:
        RI = RI[RI >= maxes[i+1][0]]

    numPhoton = RI*norm*float(tE[i])/1e12/E
    photonNoise = np.sqrt(numPhoton)
    darkNoise = DNR*float(tE[i])/1e6
    Err = np.sqrt(RD**2+photonNoise**2+darkNoise**2)

    RT+= list(RI)
    ErrR+=list(Err)

#stitch together the left and right halves
T = LT + RT
Err = ErrL+ErrR
x = np.arange(len(T)) * pxSize

peakElNum = norm*float(tE[0])/E*QE/1e12
normedERR = np.array(Err) / peakElNum

#plot measured profile with error bars
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.errorbar(x - x[np.argmax(T)], T, normedERR,fmt='o', c=(0, 176/255, 80/255, 1), lw=1.9, markersize=6)


#plot perfect gaussian profile for comparison
def gauss(x, A, a, x0):
    return A*np.exp(-a*(x-x0)**2)

params, cov = fit(gauss, x - x[np.argmax(T)], T, p0=[1, 1, 0])
x = np.linspace(-16, 16, 1000)
plt.plot(x, gauss(x, *params), '--', c='k', lw=2)
    
plt.yscale('log')
plt.ylabel("Intensity (A.U.)", size=15)
plt.xlabel("x ($\mu m$)", size=15)
plt.tick_params(labelsize=15)
plt.xlim(-16, 16)
plt.ylim(4e-6, 2)
ax.set_xticks(np.arange(-16, 17, 4))
plt.tick_params(labelsize=15)
plt.grid(True, which='both')
plt.show()



