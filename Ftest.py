# from tkinter import *
# from tkinter import ttk
# from tkinter import filedialog
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import axes3d
from scipy.fftpack import fft, ifft
from scipy.integrate import quad
from scipy import signal
import csv
import numpy as np
import seaborn as sns


# FUNCTION - Magnitudes
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
	magG = 250*(normG/32768)            #This is deg/s if range +/-250 deg/s
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


#FUNCTION - Cummulative Energy Spent from Acceleration
###############################################################################
def energy_sum(accelMag):
    E = 0.5 *  accelMag ** 2 #kinetic energy per unit mass
    CE = np.cumsum(E)        #cummulative energy
    return CE,E
###############################################################################


novicepath = 'novice_arduino.csv'     #names of csv files
expertpath = 'expert_arduino.csv'

novice = np.loadtxt(novicepath, delimiter=',', unpack=False) #load data
expert = np.loadtxt(expertpath, delimiter=',', unpack=False)
NmagA,NmagG,Nmag_ax,Nmag_ay,Nmag_az,Nmag_gx,Nmag_gy,Nmag_gz = Magnitudes(novice)
EmagA,EmagG,Emag_ax,Emag_ay,Emag_az,Emag_gx,Emag_gy,Emag_gz = Magnitudes(expert)
tN = T(novice)
tE = T(expert)

CNE,NE = energy_sum(NmagA)
CEE,EE = energy_sum(EmagA)

plt.figure(1)
plt.plot(tN,CNE,label='Novice')
plt.plot(tE,CEE,label='Expert')
plt.title('Cummulative Motion Energy per Unit Mass (J/kg)')
plt.ylabel('Total Energy (J/kg)')
plt.xlabel('Time (s)')
plt.legend()

plt.figure(2)
plt.plot(tN,NE,label='Novice')
plt.plot(tE,EE,label='Expert')
plt.title('Kinetic Energy per Unit Mass (J/kg)')
plt.ylabel('Kinetic Energy (J/kg)')
plt.xlabel('Time (s)')
plt.legend()
plt.show()
