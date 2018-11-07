# from tkinter import *
# from tkinter import ttk
# from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from scipy.fftpack import fft, ifft
from scipy import signal
import csv
import numpy as np
import seaborn as sns

# #FUNCTION - Animate Graphs building in real time
# ###############################################################################
# def animate(i):
# 	timesteps = len(accelData)
# 	t = np.arange(0,timesteps*0.2,0.2)  #5 Hz
# #	t = np.arange(0,timesteps,0.005556)  #180 Hz
# 	return t
# ###############################################################################


#FUNCTION - Magnitudes
# Normalises the Magnitudes of the Accelerometer and Gyroscope
###############################################################################
def Magnitudes(accelData):
	A = np.matrix(accelData[:,0:3]) #Assign 1st 3 columns of data to A
	G = np.matrix(accelData[:,3:6]) #Assign 2nd 3 columns of data to G
	ax= np.array(accelData[:,0])
	ay= np.array(accelData[:,1])
	az= np.array(accelData[:,2])
	gx= np.array(accelData[:,3])
	gy= np.array(accelData[:,4])
	gz= np.array(accelData[:,5])

	normA = np.linalg.norm(A,axis=1,keepdims=True) #This calculates the magnitude for acceleration
	normG = np.linalg.norm(G,axis=1,keepdims=True) #This calculates the magnitude for rotation

	magA = (normA/16384)-1              #This is 1g if using sensor range +/- 2g
	magG = 250*(normG/32768)        #This is deg/s if range +/-250 deg/s
	mag_ax = (ax/16384)-1
	mag_ay = (ax/16384)-1
	mag_az = (ax/16384)-1
	mag_gx = 250*gx/32768
	mag_gy = 250*gx/32768
	mag_gz = 250*gx/32768


	return magA,magG,mag_ax,mag_ay,mag_az,mag_gx,mag_gy,mag_gz
###############################################################################

#FUNCTION - Create time vectors for plotting based on sampling rates
###############################################################################
def T(accelData):
	timesteps = len(accelData)
	# t = np.arange(0,timesteps*0.2,0.2)  #5 Hz
	t = np.arange(0,timesteps/180,(1/180))  #180 Hz
	return t
###############################################################################

#FUNCTION - deltas
###############################################################################
def deltas(magVector):
	A = np.array(magVector[0:len(magVector)-1])
	B = np.array(magVector[1:len(magVector)])
	deltaVector = ((A-B)**2)**0.5
	return deltaVector
###############################################################################

#FUNCTION - Moving average
###############################################################################
def movingAvg(values,window):
	weights = np.repeat(1.0, window)/window
	ma = np.convolve(values,weights,'valid')
	return ma
###############################################################################



#FUNCTION - Movements
###############################################################################
def Movements(MAData,gThreshold):

	Moving = MAData > gThreshold
	totalMoves = 0

	for i in range(len(Moving[:-1])):
		if Moving[i] == True and Moving[i+1] == False:
	 		totalMoves = totalMoves + 1

	return totalMoves, Moving

#FUNCTION - RangeMovements
###############################################################################
def RangeMovements(MAData):
	RThreshold = [0.002,0.004,0.006,0.008,0.010,0.020,0.030,0.040]
	RtotalMoves = [0,0,0,0,0,0,0,0]
	for j in range(len(RThreshold)):
		Moving = MAData > RThreshold[j]
		RtotalMoves[j] = 0
		for i in range(len(Moving[:-1])):
			if Moving[i] == True and Moving[i+1] == False:
				RtotalMoves[j] = RtotalMoves[j] + 1

	return RtotalMoves

#FUNCTION - FFTAnalysis
###############################################################################
def FFTAnalysis(accelData,Fs):
	# Number of sample points
	N = len(accelData)
	# sample spacing
	T = 1.0 / Fs #sample spacing based on sample rate (Hz)
	t = np.linspace(0.0, N*T, N) #time vector
	yf = fft(accelData)          #fft of data
	P2 = abs(yf/N)
	P1 = P2[0:N//2+1]
	P1[1:-2] = 2*P1[1:-2]
	tf = np.linspace(0.0, 1.0/(2.0*T), N//2)

	return tf,yf,N

###############################################################################
#    MAIN CODE
#Load in the test data and run the functions to produce the output plots
###############################################################################

# novicepath = 'novice_arduino.csv'     #names of csv files
# expertpath = 'expert_arduino.csv'

novice = np.loadtxt(novicepath, delimiter=',', unpack=False) #load data
expert = np.loadtxt(expertpath, delimiter=',', unpack=False)

NmagA,NmagG,Nmag_ax,Nmag_ay,Nmag_az,Nmag_gx,Nmag_gy,Nmag_gz = Magnitudes(novice)  #convert raw data to vector magnitudes
EmagA,EmagG,Emag_ax,Emag_ay,Emag_az,Emag_gx,Emag_gy,Emag_gz = Magnitudes(expert)

tN = T(novice)  #create the time vectors from the data
tE = T(expert)

dNA = deltas(NmagA)
dNG = deltas(NmagG)
dEA = deltas(EmagA)
dEG = deltas(EmagG)

plt.figure(1)
sns.distplot(dNA, bins=20, kde=False, rug=True)

# print(np.shape(NmagA[:,0]))
NMAData = movingAvg((dNA[:,0]),1)
EMAData = movingAvg((dEA[:,0]),1)

# NMAData2 = movingAvg((NmagA[:,0]),1000)
# NMAData3 = movingAvg((NmagA[:,0]),2000)
# NMAData4 = movingAvg((NmagA[:,0]),5000)
#
plt.figure(2)
plt.plot(NmagA,label='Novice')
plt.plot(EmagA,label='Expert')
plt.legend()
# plt.plot(NMAData2)
# plt.plot(NMAData3)
# plt.plot(NMAData4)

movesNovice, MovingN = Movements(NMAData,0.008)
movesExpert, MovingE = Movements(EMAData,0.008)
print('novice moves', movesNovice, '\n')
print('expert moves', movesExpert, '\n')

RmovesNovice = RangeMovements(NMAData)
RmovesExpert = RangeMovements(EMAData)
print('All novice moves', RmovesNovice, '\n')
print('All expert moves', RmovesExpert, '\n')

plt.figure(5)
plt.bar([1,2,3,4,5,6,7,8],RmovesNovice)
plt.bar([1,2,3,4,5,6,7,8],RmovesExpert)



Ntf,Nyf,NN = FFTAnalysis(NmagA,180)
Etf,Eyf,EN = FFTAnalysis(EmagA,180)
plt.figure(3)
plt.plot(Ntf, 2.0/NN * np.abs(Nyf[0:NN//2]),label='Novice')
plt.plot(Etf, 2.0/EN * np.abs(Eyf[0:EN//2]),label='Expert')
plt.grid()
plt.title('Single-Sided Amplitude Spectrum of y(t)')
plt.ylabel('|Y(f)|')
plt.xlabel('Frequency (Hz)')
# plt.axis([0, 20,0,0.00015])
plt.legend()




# sns.set(style="ticks")
#
# rs = np.random.RandomState(11)
# x = rs.gamma(2, size=1000)
# y = -.5 * x + rs.normal(size=1000)
#
# sns.jointplot(RmovesNovice, RmovesExpert, kind="hex", color="#4CB391")
# sns.jointplot(x, y, kind="hex", color="#4CB391")

# widths = np.arange(np.shape(NmagA)[])
# cwtmatr = signal.cwt(NmagA,signal.ricker,widths)
# plt.figure(3)
# plt.plot(cwtmatr,label='Novice')
# plt.legend()

# plt.figure(1)
# plt.subplot(211)
# plt.plot(tN,NmagA[0:len(tN)])  #plot the acceleration vs time for the novice
# plt.plot(tE,EmagA[0:len(tE)])  #plot the Expert data on top
# plt.ylabel('Acceleration g (m/s^2)')
# # plt.xlabel('Time (s)')
#
# plt.subplot(212)
# plt.plot(tN,NmagG[0:len(tN)])  #plot the acceleration vs time for the novice
# plt.plot(tE,EmagG[0:len(tE)])  #plot the Expert data on top
# plt.ylabel('Ang Vel Mag (deg/s)')
# plt.xlabel('Time (s)')
#
# plt.figure(2)
# plt.subplot(211)
# plt.plot(tN[0:-1],dNA)  #plot the acceleration vs time for the novice
# plt.plot(tE[0:-1],dEA)  #plot the Expert data on top
# plt.ylabel('delta Acceleration g (m/s^2)')
# # plt.xlabel('Time (s)')
#
# plt.subplot(212)
# plt.plot(tN[0:-1],dNG)  #plot the acceleration vs time for the novice
# plt.plot(tE[0:-1],dEG)  #plot the Expert data on top
# plt.ylabel('delta Ang Vel Mag (deg/s)')
# plt.xlabel('Time (s)')

plt.show()
