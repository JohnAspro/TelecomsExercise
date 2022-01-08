import os
import time
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from scipy import signal
from matplotlib.ticker import StrMethodFormatter

# School ID = 03118942
# F=9+4+2=15=>1+5=6

myFreq=6000
A=4

# Βοηθιτικες Συναρτησεις

def PlotYLim(Max, Min):
	plt.ylim([Min,Max])

def plotSignals(time1, signal1, color1, legend1, PlotTitle, numberOfSignals=1, time2=None, signal2=None, color2=None, legend2=None):
	if numberOfSignals==1:
		plt.plot(time1, signal1, color1)
		plt.legend(legend1)
	elif numberOfSignals==2:
		plt.plot(time1, signal1, color1, time2, signal2, '.', color2)
		plt.legend([legend1, legend2])
	else:
		return None
	plt.xlabel('Seconds')
	plt.ylabel('Volts')
	plt.title(PlotTitle)
	plt.grid()
	plt.show()

#---------------|ΑΣΚΗΣΗ 2|-------------------

#(A)
fs1=30*myFreq #180kHz
fs2=50*myFreq #300kHz


def mid_riser(signal):
	for i in range(len(signal)):
		if signal[i]>0xb0111:
			signal[i]=7
		elif signal[i]<-0xb1000:
			signal[i]=-8
		else:
			if (signal[i] - round(signal[i]) > 0) and (signal[i] > 0):
				signal[i] = round(signal[i]) + 1
			elif (signal[i] - round(signal[i]) < 0) and (signal[i] < 0):
				signal[i] = round(signal[i]) - 1
			else:
				signal[i] = round(signal[i])
	return signal

# grayCodeBinary = [0000, 0001, 0011, 0010, 0110, 0111, 0101, 0100, 1100, 1101, 1111, 1110, 1010, 1011, 1001, 1000]

def grayCodeMap(signal):
	grayCode4bit = [0, 1, 3, 2, 6, 7, 5, 4, 12, 13, 15, 14, 10, 11, 9, 8] 
	for i in range(len(signal)):
		signal[i] = grayCode4bit[int(signal[i])+8]
	return signal

def calcError(QuantifiedSamples, accualSignalSamples, numOfSamples):
	i=0
	s=0
	while i < numOfSamples:
		s+=accualSignalSamples[i]-QuantifiedSamples[i]
		i+=1
	return s/numOfSamples

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
# t1 = np.linspace(0, 4/myFreq, 4*int(fs1/myFreq))
t1 = np.arange(0, 4/myFreq, 1/fs1)
triangle1 = signal.sawtooth(2 * np.pi * myFreq * t1, 0.5)*4
trigCopy = signal.sawtooth(2 * np.pi * myFreq * t1, 0.5)
x = mid_riser(triangle1)

# y = grayCodeMap(x)
fig, ax = plt.subplots()

ax.yaxis.set_major_formatter(StrMethodFormatter("{x:04b}"))
ax.yaxis.set_ticks(np.arange(-4, 15, 1))

plotSignals(t1, 4*trigCopy, 'o', 'Fs1', 'Quantified Triangle sampled Fs1')
plotSignals(t1, x, 'o', 'Fs1', 'Quantified Triangle sampled Fs1')
plt.show()

print(calcError(mid_riser(triangle1), trigCopy, 10))
print(calcError(mid_riser(triangle1), trigCopy, 20))

# print(calcSNR(4*triangle1, 10))
# print(calcSNR(4*triangle1, 20))