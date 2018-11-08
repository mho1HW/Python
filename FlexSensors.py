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


#FUNCTION - Fingers (Normalise Flex Data)
###############################################################################
def fingers(flexData):
    indexF = np.array(flexData[:,0]) #1st column Index Finger
    middleF = np.array(flexData[:,1]) #2nd column Middle Finger
    indexFlex = (indexF-600)/400 #normalised from raw flex range of 600 to 1000
    middleFlex = (middleF-600)/400 #normalised from raw flex range of 600 to 1000
    return indexFlex, middleFlex

#FUNCTION - Fmovements - Calculates Finger Movements
###############################################################################
def Fmovements(fFlex,FlexNum):
    totalFMoves = 0
    boolFlex = fFlex > FlexNum

    for i in range(len(fFlex[:-1])):
        if boolFlex[i] == True and boolFlex[i+1] == False:
            totalFMoves = totalFMoves + 1

    return totalFMoves

#FUNCTION - FNumMoves- Total Range and Number of Movements
###############################################################################
def FNumMoves(fFlex):
    Rmax = max(fFlex)
    Rmin = min(fFlex)
    RangeMoves = [0,0,0,0,0,0,0,0,0,0]
    moveInt = np.linspace(Rmin,Rmax,10)
    for i in range(10):
        RangeMoves[i] = Fmovements(fFlex,moveInt[i])

    return RangeMoves,Rmax,Rmin

###############################################################################
#    MAIN CODE
#Load in the test data and run the functions to produce the output plots
###############################################################################

noviceFlex = 'Flex Sensor Novice.csv'     #names of csv files
expertFlex = 'Flex Sensor Expert.csv'

novice = np.loadtxt(noviceFlex, delimiter=',', unpack=False) #load data
expert = np.loadtxt(expertFlex, delimiter=',', unpack=False)

NiF,NmF = fingers(novice)
EiF,EmF = fingers(expert)

Fmoves_Ni,NiMax,NiMin = FNumMoves(NiF)
Fmoves_Nm,NmMax,NmMin = FNumMoves(NmF)
Fmoves_Ei,EiMax,EiMin = FNumMoves(EiF)
Fmoves_Em,EmMax,EmMin = FNumMoves(EmF)
print(Fmoves_Ni,NiMin,NiMax)
print(Fmoves_Nm,NmMin,NmMax)
print(Fmoves_Ei,EiMin,EiMax)
print(Fmoves_Em,EmMin,EmMax)

plt.figure(1)
plt.plot(NiF,label='Novice Index')
plt.plot(NmF,label='Novice Middle')
plt.plot(EiF,label='Expert Index')
plt.plot(EmF,label='Expert Middle')
plt.title('Normalised Finger Flex')
plt.ylabel('Finger Flex (-)')
plt.xlabel('Data Point')
plt.legend()

plt.figure(2)
plt.plot(Fmoves_Ni,label='Novice Index')
plt.plot(Fmoves_Nm,label='Novice Middle')
plt.plot(Fmoves_Ei,label='Expert Index')
plt.plot(Fmoves_Em,label='Expert Middle')
plt.title('Number of Finger Movements')
plt.ylabel('Number')
plt.xlabel('Range Interval')
plt.legend()
plt.show()
