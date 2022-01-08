import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal
from matplotlib.ticker import StrMethodFormatter
import math
import statistics

# School ID = 03118942
# F=9+4+2=15=>1+5=6

OurFreqs=[6000, 8000]
myFreq = OurFreqs[0]
A=4

#---------------|ΑΣΚΗΣΗ 1|-------------------

#-----------Βοηθητικες Συναρτησεις------------

def PlotYLim(Max, Min):
	plt.ylim([Min,Max])

def plotSignals(time1, signal1, color1, legend1, PlotTitle, freq=myFreq, numberOfSignals=1, time2=None, signal2=None, color2=None, legend2=None):
	if numberOfSignals==1:
		plt.plot(time1, signal1, color1)
		plt.legend([legend1])
	elif numberOfSignals==2:
		plt.plot(time1, signal1, color1, time2, signal2, color2)
		plt.legend([legend1, legend2])
	else:
		return None
	numOfPeriods = time1[-1]*freq
	plt.xticks(np.arange(0, (numOfPeriods+0.5)/freq, 1/(2*freq)))
	plt.xlabel('Time [s]')
	plt.ylabel('Amplitude [V]')
	plt.title(PlotTitle)
	plt.grid()
	plt.show()

#(A)
fs1=30*myFreq #180kHz
fs2=50*myFreq #300kHz

t1 = np.arange(0, 4/myFreq+1/fs1, 1/fs1)
t2 = np.arange(0, 4/myFreq+1/fs2, 1/fs2)
triangle1 = signal.sawtooth(2 * np.pi * myFreq * t1, 0.5)*4
triangle2 = signal.sawtooth(2 * np.pi * myFreq * t2, 0.5)*4

plotSignals(t1, triangle1, 'o', 'Fs1', 'Fs1 Sampling Rate')

plotSignals(t2, triangle2, 'o', 'Fs2', 'Fs2 Sampling Rate')

#ploting Both
plotSignals(t1, triangle1, 'o', 'Fs1', 'Both Sampling Rates', myFreq, 2, t2, triangle2, '.', 'Fs2')

#(B) 
fs3 = 4*myFreq #24kHz
t3 = np.arange(0, 4/myFreq+1/fs3, 1/fs3)
triangle3 = signal.sawtooth(2 * np.pi * myFreq * t3, 0.5)*4

plotSignals(t3, triangle3, 'o', 'Fs3', 'Fs3 Sampling Rate')

#(C)

#(i)
def getSin(time):
	return np.sin(np.pi*2*myFreq*time)

#(a)
plotSignals(t1, getSin(t1), 'o', 'Fs1', 'Sin Sampling Rate Fs1')

plotSignals(t2, getSin(t2), 'o', 'Fs2', 'Sin Sampling Rate Fs2')

plotSignals(t1, getSin(t1), 'o', 'Fs1', 'Sin Both Sampling Rates', myFreq, 2, t2, getSin(t2), '.', 'Fs2')

#(b)
plotSignals(t3, getSin(t3), 'o', 'Fs3', 'Sin Sampling Rate Fs3')

#(ii)
addedFreq = 1000 #1 kHz
fc = math.gcd(myFreq, myFreq + addedFreq) # GCD(6κ, 7κ)=>1k


def getQ(time):
	return getSin(time) + np.sin(np.pi*2*(myFreq+addedFreq)*time)

def getQtimeline(SamplingFreq):
	return np.arange(0, 1/addedFreq+1/SamplingFreq, 1/SamplingFreq)

#(a)
plotSignals(getQtimeline(fs1), getQ(getQtimeline(fs1)), 'o', 'Fs1', 'Q Signal Sampling Rate Fs1', fc)

plotSignals(getQtimeline(fs2), getQ(getQtimeline(fs2)), 'o', 'Fs1', 'Q Signal Sampling Rate Fs1', fc)

plotSignals(getQtimeline(fs1), getQ(getQtimeline(fs1)), 'o', 'Fs1', 'Q Signal Both Sampling Rates', fc, 2, getQtimeline(fs2), getQ(getQtimeline(fs2)), '.', 'Fs2')

#(b)
plotSignals(getQtimeline(fs3), getQ(getQtimeline(fs3)), 'o', 'Fs3', 'Q Signal Sampling Rate Fs3', fc)


#---------------|ΑΣΚΗΣΗ 2|-------------------

# Θα χρεισταστουμε την συνχοτητα fs1 και το t1 του πρωτου ερωτηματος

def getQuantLevel(index):
	return (index-1)*0.5+0.25

def mid_riser(signal):
	for i in range(len(signal)):
		if signal[i]>3.5: #7
			signal[i]=3.75
		elif signal[i]<-3.5: #-8
			signal[i]=-3.75
		else:
			if signal[i] > 0:
				signal[i] = getQuantLevel(int(signal[i]/0.5)+1)
			else: 
				signal[i] = -getQuantLevel(abs(int(signal[i]/0.5))+1)
	return signal

# Κοιταμε την μεση αποκλιση των κβαντισμενων δειγματων σε σχεση με την αρχικη συναρτηση τριγωνου
# Επισης ο χρηστης μπορει να βαλει τον αριθμο δειγματων για ελεγχο
def calcError(QuantifiedSamples, accualSignalSamples, numOfSamples):
	i=0
	s = [None] * numOfSamples
	while i < numOfSamples:
		s[i]=accualSignalSamples[i]-QuantifiedSamples[i]
		i+=1
	return statistics.stdev(s)

def calcAverageSigPower(signal, numOfSamples):
	i=0
	s=0
	while i < numOfSamples:
		s += signal[i]**2
	return s/numOfSamples

def calcSNR(StartingSignal, numOfSamples):
	numOfBitsPerSample = 4
	maxSigVoltage = 7
	return ((2**(2*numOfBitsPerSample))*(3*calcAverageSigPower(StartingSignal, numOfSamples)/maxSigVoltage**2))

#(a)
fig, ax = plt.subplots()

grayCodeBinary = ["0000", "0001", "0011", "0010", "0110", "0111", "0101", "0100", "1100", "1101", "1111", "1110", "1010", "1011", "1001", "1000"]
quantValues = [-3.75, -3.25, -2.75, -2.25, -1.75, -1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75]
ax.set_yticks(quantValues)
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:04b}"))
ax.set_yticklabels(grayCodeBinary)

QuantifiedSig = mid_riser(signal.sawtooth(2 * np.pi * myFreq * t1, 0.5)*4)

plotSignals(t1, QuantifiedSig, 'o', 'Fs1', 'Quantified Triangle sampled Fs1', myFreq)

#(b)
#(i)
triangle1 = signal.sawtooth(2 * np.pi * myFreq * t1, 0.5)*4
print(calcError(mid_riser(signal.sawtooth(2 * np.pi * myFreq * t1, 0.5)*4), triangle1, 10))
#(ii)
print(calcError(mid_riser(signal.sawtooth(2 * np.pi * myFreq * t1, 0.5)*4), triangle1, 20))
#(iii)
# print(calcSNR(triangle1, 10))
# print(calcSNR(triangle1, 20))