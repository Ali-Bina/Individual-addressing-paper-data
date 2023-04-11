import numpy as np, matplotlib.pyplot as plt
import pandas as pd
import glob
import random


#0.1 ms data for all 16 channels
files01 = glob.glob("*_0.1ms*.xlsx")
files01 = sorted(files01, key=lambda x: int(x.split("_")[0]))


#5 ms data for all 16 channels
files5 = glob.glob("*_5ms*.xlsx")
files5 = sorted(files5, key=lambda x: int(x.split("_")[0]))

#average background 
bgAvg = np.array([2.252996550665955, 17.599242422315868]) 

#Calibration Slopes calculated from signal vs power plots for each exposure time
calib = np.array([37516.24496574,1064903.089660743]) #adu/W

#The exposure times used for taking the profiles
tE = [0.1, 5] #ms

#camera pixel size
pxsize = 3.2 #um

#read the excel files containing the profile data and store them in a list
data01 = []
for i, file in enumerate(files01):
    data = pd.read_excel(file, header=3)
    data = np.array(data)
    data01.append(data)
    print("File number:", i+1)

print("___________________")
print("onto the 5 ms data")
data5 = []
for i, file in enumerate(files5):
    data = pd.read_excel(file, header=3)
    data = np.array(data)
    data5.append(data)
    print("File number:", i+1)
   


N = len(files01) #number of channels
M = data01[0].shape[1] #number of pixels 
colors=[(random.random(), random.random(), random.random()) for i in range(N)]
colors=plt.cm.winter(np.linspace(0,1,N))

#account for slight offsets in the relative positions of the 0.1 and 5 ms data
offSet = np.array([0, 0, 0, 0, 3.14, 7.059, 0, 3.149, 0, 0, 4, 0, 0, 3, 4, 0])


# photon energy
E = 6.626e-34 * 2.998e8/(532e-9) #J

#couldn't find these values for the camera so I assumed the upperbounds given on
#https://www.microscopyu.com/tutorials/ccd-signal-to-noise-ratio#:~:text=Sliders%20are%20provided%20for%20varying,electrons%20per%20pixel%20per%20second).
#typical values for both CCD and CMOS are smaller than these upper bounds
DNR = 50 #electrons per pixel per second
RD = 20 #rms electrons per pixel

#Quantum efficiency of 0.6-0.7 is typical for cmos:
#https://scientificimaging.com/knowledge-base/qe-curves-for-cmos-imagers/
#https://www.flir.com/products/blackfly-s-usb3/?model=BFS-U3-04S2M-C&vertical=machine+vision&segment=iis
QE = 0.6 


for i in range(N):
    
    #calibrate and then plot 0.1 ms data

    data = data01[i]

    #subtract background and apply calibration to convert ADU to intensity
    data = (data - bgAvg[0])/calib[0]

    #find peak of profile
    maxInd = np.unravel_index(np.argmax(data), data.shape)
    xmax, ymax = maxInd
    maxVal = data[xmax, ymax]

    #normalize intensity to peak value
    data /= maxVal

    x = np.array(range(M))*pxsize
    xval = x[ymax] #x position of peak

    #trim 0.1 ms data near noise floor to stitch with 5 ms exposure data
    x = x[data[xmax, :] > 1e-2]
    data1 = data[xmax, data[xmax, :] > 1e-2]

    #error analysis
    numPhoton = data1*maxVal*float(tE[0])/1e9/E
    photonNoise = np.sqrt(numPhoton)
    darkNoise = DNR*float(tE[0])/1e3
    Err = np.sqrt(RD**2+photonNoise**2+darkNoise**2)
    peakElNum = maxVal*float(tE[0])/E*QE/1e9
    normedERR = np.array(Err) / peakElNum   

    plt.errorbar(x - offSet[i], data1, yerr=normedERR,lw=2, c=(0, 176/255, 80/255, 1))

    #Calibrate and then plot 5 ms data

    data = data5[i]
    data = (data - bgAvg[1])/calib[1]
    x = np.array(range(M))*pxsize

    data /= maxVal

    #trim the noise floor from plot
    x = x[data[xmax, :] > 1e-5]
    data2 = data[xmax, data[xmax, :] > 1e-5]

    #Trim off noise floor for channel 1 so it doesn't extend across all others
    if i == 0:
        data2 = data2[x<500]
        x = x[x<500]

    #getting rid of the saturated top of the 5 ms data
    d2Max = np.max(data2)/1.5
    x = x[data2<d2Max]
    data2 = data2[data2<d2Max]
   
    #Left half of 5 ms data
    data2_1 = data2[x<xval]
    x_1 = x[x<xval]

    #right half of data
    data2_2 = data2[x>xval]
    x_2 = x[x>xval]
   

    #error analysis
    numPhoton = data2_1*maxVal*float(tE[1])/1e9/E
    photonNoise = np.sqrt(numPhoton)
    darkNoise = DNR*float(tE[1])/1e3
    Err = np.sqrt(RD**2+photonNoise**2+darkNoise**2)
    peakElNum = maxVal*float(tE[0])/E*QE/1e9
    normedERR_1 = np.array(Err) / peakElNum 

    numPhoton = data2_2*maxVal*float(tE[1])/1e9/E
    photonNoise = np.sqrt(numPhoton) 
    Err = np.sqrt(RD**2+photonNoise**2+darkNoise**2)
    normedERR_2 = np.array(Err) / peakElNum 

    #plot both halves
    #The halves were plotted separately to get rid of the line that would have otherwise connected them during plotting
    plt.errorbar(x_1, data2_1, yerr=normedERR_1, lw=2, c=(0, 176/255, 80/255, 1))
    plt.errorbar(x_2, data2_2, yerr=normedERR_2, lw=2, c=(0, 176/255, 80/255, 1))
   
    plt.yscale("log")
    plt.ylim(5e-5, 1)
    plt.xlim(0, 2201)
    plt.xlabel("x ($\mu$m)", size=20)
    plt.tick_params(labelsize=20)
    plt.ylabel("Intensity (norm.)", size=20)
    plt.grid(True, which='both')
plt.show()


#crossTalkAnalysis

#stores calcualted cross talk value for channel i and its estimated error
crossTs = []
Errs = []

#For each channel, average the intensity of the tail of its two neighbours located at the peak of the channel 
for i in range(N):
   
    #load in 0.1 ms data to find peak position of channel i
    data = data01[i]
    data = (data - bgAvg[0])/calib[0]
    maxInd = np.unravel_index(np.argmax(data), data.shape)
    xmax, ymax = maxInd
    maxVal = data[xmax, ymax]
    x = np.array(range(M))*pxsize
    xval = x[ymax] #peak position of channel i
    

    #Store index of neighbours of channel i in a list
    cross = 0 #sum of cross-talk value from neighbours
    Terr = 0
    if i == 0:
        neib = [1]
    elif i == N-1:
        neib = [N-2]
    else:
        neib = [i - 1, i + 1]

    for j in neib:

            #load in data for neighbours of i
            data = data01[j]
            data = (data - bgAvg[0])/calib[0]
            maxInd = np.unravel_index(np.argmax(data), data.shape)
            xmax, ymax = maxInd
            maxVal = data[xmax, ymax]

            #tails of the 5 ms data overlap with channel i and cause cross-talk
            data = data5[j]
            data = (data[xmax,:] - bgAvg[1])/calib[1]
            data /= maxVal

            #average over 6 um around the peak of channel i
            crossAvg = data[np.abs(x-xval)<6]
            
            #error analysis
            numPhoton = crossAvg*maxVal*float(tE[1])/1e9/E
            photonNoise = np.sqrt(numPhoton)
            darkNoise = DNR*float(tE[1])/1e3
            Err = np.sqrt(RD**2+photonNoise**2+darkNoise**2)
            peakElNum = maxVal*float(tE[0])/E*QE/1e9
            normedERR = np.array(Err) / peakElNum  
            normedERR = normedERR[~np.isnan(normedERR)]

            #ignore values within the noise floor
            Err = Err[crossAvg>1e-5]
            crossAvg = crossAvg[crossAvg>1e-5]
           
            #add up contribution from the right and left channels
            if crossAvg.size > 0:
                cross += np.average(crossAvg)
                Terr +=  np.sum(normedERR**2)/len(normedERR)**2

    crossTs.append(cross)
    Errs.append(np.sqrt(Terr))

Errs = np.array(Errs)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.bar(range(1, N+1), np.array(crossTs) * 1e4, yerr=Errs*1e4, capsize=2, color='royalblue')
plt.xlabel("Channel", size=18)
plt.ylabel(r"Intensity Crosstalk ($\times$ 10$^{-4}$)", size=18)
plt.tick_params(labelsize=18)
ax.set_xticks(np.arange(1, 17, 1))
plt.show()
