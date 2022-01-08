import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal
import math

# School ID = 03118942
# F=9+4+2=15=>1+5=6

myFreq=8000
A=4

#-----------Βοηθητικες Συναρτησεις------------

def PlotYLim(Max, Min):
	plt.ylim([Min,Max])

def plotSignals(time1, signal1, color1, legend1, PlotTitle, numberOfSignals, freq, numOfPeriods, time2=None, signal2=None, color2=None, legend2=None):
	if numberOfSignals==1:
		plt.plot(time1, signal1, color1)
		plt.legend(legend1)
	elif numberOfSignals==2:
		plt.plot(time1, signal1, color1, time2, signal2, color2)
		plt.legend([legend1, legend2])
	else:
		return None
	plt.xticks(np.arange(0, numOfPeriods/freq, 1/(2*freq)))
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude [V]')
	plt.title(PlotTitle)
	plt.grid()
	plt.show()

#---------------|ΑΣΚΗΣΗ 1|-------------------

#(A)
fs1=30*myFreq #180kHz
fs2=50*myFreq #300kHz

t1 = np.arange(0, 4/myFreq, 1/fs1)
t2 = np.arange(0, 4/myFreq, 1/fs2)
triangle1 = 4*signal.sawtooth(2 * np.pi * myFreq * t1, 0.5)
triangle2 = 4*signal.sawtooth(2 * np.pi * myFreq * t2, 0.5)

plotSignals(t1, triangle1, 'o', 'Fs1', 'Fs1 Sampling Rate', 1, myFreq, 4.5)

plotSignals(t2, triangle2, 'o', 'Fs2', 'Fs2 Sampling Rate', 1, myFreq, 4.5)

#ploting Both
plotSignals(t1, triangle1, 'o', 'Fs1', 'Both Sampling Rates', 2, myFreq, 4.5, t2, triangle2, 'o', 'Fs2')

#(B)
fs3 = 4*myFreq #24kHz
t3 = np.arange(0, 4.25/myFreq, 1/fs3)
triangle3 = 4*signal.sawtooth(2 * np.pi * myFreq * t3, 0.5)
plotSignals(t3, triangle3, 'o', 'Fs3', 'Fs3 Sampling Rate', 1, myFreq, 4.5)

#(C)

#(i)
def getSin(time):
	return np.sin(np.pi*2*myFreq*time)

#(a)
plotSignals(t1, getSin(t1), 'o', 'Fs1', 'Sin Sampling Rate Fs1', 1, myFreq, 4.5)

plotSignals(t2, getSin(t2), 'o', 'Fs2', 'Sin Sampling Rate Fs2', 1, myFreq, 4.5)

plotSignals(t1, getSin(t1), 'o', 'Fs1', 'Sin Both Sampling Rates', 2, myFreq, 4.5, t2, getSin(t2), 'o', 'Fs2')

#(b)
plotSignals(t3, getSin(t3), 'o', 'Fs3', 'Sin Sampling Rate Fs3', 1, myFreq, 4.5)

#(ii)
addedFreq=1000 #1 kHz

def getQ(time):
	return getSin(time) + np.sin(np.pi*2*(myFreq+addedFreq)*time)

#(a)
fc = math.gcd(myFreq, myFreq + addedFreq) # κοινή συχνότητα
t1 = np.arange(0, 1/fc, 1/fs1)
t2 = np.arange(0, 1/fc, 1/fs2)
t3 = np.arange(0, 1/fc, 1/fs3)

plotSignals(t1, getQ(t1), 'o', 'Fs1', 'Q Signal Sampling Rate Fs1', 1, fc, 1.5)

plotSignals(t2, getQ(t2), 'o', 'Fs1', 'Q Signal Sampling Rate Fs1', 1, fc, 1.5)

plotSignals(t1, getQ(t1), 'o', 'Fs1', 'Q Signal Both Sampling Rates', 2, fc, 1.5, t2, getQ(t2), 'o', 'Fs2')

#(b)
plotSignals(t3, getQ(t3), 'o', 'Fs3', 'Q Signal Sampling Rate Fs3', 1, fc, 1.5)


#---------------|ΑΣΚΗΣΗ 2|-------------------

# Θα χρεισταστουμε την συνχοτητα fs1 και το t1 του πρωτου ερωτηματος

fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(StrMethodFormatter("{x:04b}"))
