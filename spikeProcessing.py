#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy
import pylab
import nest.raster_plot as raster
from scipy import signal as sig

#filePath = 'oksperiences_02_13-14_modelNo9_x10_PD/2017_2_14_15:5_00000/log/' # GSTN*1.8, GGPe*1.8
filePath = '2017_3_23_14:26_00000/log/' # GSTN*1.8, GGPe*1.8
#filePath = '2017_2_13_14:42_00000/log/' # GSTN*1.5, GGPe*1.5
#filePath = '2017_2_14_11:20_00000/log/' # normal

NUCLEI = ['MSN','FSI','STN','GPe','GPi']
fileList = {}

rootFID = 71625

i = 0
for N in NUCLEI:
  fileList[N] = [N+'-'+str(rootFID+i)+'-0.gdf',N+'-'+str(rootFID+i)+'-1.gdf']
  i += 1

#fileList['GPe'] = ['GPe-71628-0.gdf','GPe-71628-1.gdf']
#fileList['FSI'] = ['FSI-71627-0.gdf','STN-71627-1.gdf']
#fileList['MSN'] = ['STN-71627-0.gdf','STN-71627-1.gdf']
#fileList['GPi'] = ['STN-71627-0.gdf','STN-71627-1.gdf']

#showFFT = True

# read files & combine data :
#----------------------------
data = {}
ts = {}
gids = {}
for N in NUCLEI:
  data[N] = None
  for f in fileList[N]:
    newdata = numpy.loadtxt(filePath+f)
    if data[N] is None:
      data[N] = newdata
    else:
      data[N] = numpy.concatenate((data[N],newdata))

  ts[N] = data[N][:,1] # complete time series of spiking events
  gids[N] = data[N][:,0] # corresponding list of firing neuron ID

# prepare signal : histogram of spiking events
#---------------------------------------------
signal = {}
t_bins = numpy.arange(numpy.amin(ts[NUCLEI[0]]),numpy.amax(ts[NUCLEI[0]]),1.0)
for N in NUCLEI:
  signal[N],bins = raster._histogram(ts[N], bins=t_bins)

# show/save histogram
#---------------
pylab.rcParams["figure.figsize"] = (16,6)
ax = {}
i = 0
for N in NUCLEI:
  i += 1
  nbNeurons = len(numpy.unique(gids[N]))
  heights = 1000 * signal[N] / (1.0 * nbNeurons)
  ax[N] = pylab.subplot(320+i)  
  ax[N].bar(t_bins, heights, width=1.0, color="black")
  pylab.ylabel('Frequency [Hz]')
  pylab.xlabel('Time [sec]')

pylab.savefig('ActHisto.pdf', bbox_inches='tight')
  #pylab.show()

'''
# compute FFT
#------------
T = 0.001 #1 ms, sampling period
for N in NUCLEI:
  Nb = signal[N].size

  #ps = numpy.abs(numpy.fft.fft(signal[N]))**2
  #freqs = numpy.fft.fftfreq(Nb,T)
  #idx = numpy.argsort(freqs)
  #idx = numpy.fft.fftshift(freqs)

  yf = numpy.fft.fft(signal[N])
  xf = numpy.linspace(0.,1./(2.*T),Nb/2)

  # show FFT
  #---------
  #pylab.plot(freqs[idx], ps[idx])
  #pylab.show()

  if showFFT:
    pylab.plot(xf[1:], 2.0/Nb * numpy.abs(yf[1:Nb//2])**2)
    #pylab.plot(xf[1:], numpy.abs(yf[1:Nb//2])**2)
    pylab.savefig(N+'_PwSpec.pdf', bbox_inches='tight')
    pylab.xlabel('Freq. [Hz]')
    #pylab.show()

  # Oscillation index computation :
  #--------------------------------
  PS = 2.0/Nb * numpy.abs(yf[1:Nb//2])**2
  #print PS.size
  #print xf[15*5:30*5],PS[15*5:30*5]
  #print PS.sum()
  print N,"OI[15-30Hz]:",PS[15*5:30*5].sum()/PS.sum()

# Fano Factor computation :
# recompute histogram with 5ms bin size as Kumar
#--------------------------
signal2 = {}
t_bins = numpy.arange(numpy.amin(ts[NUCLEI[0]]),numpy.amax(ts[NUCLEI[0]]),5.0)
FF={}
for N in NUCLEI:
  signal2[N],bins = raster._histogram(ts[N], bins=t_bins)
  FF[N] = numpy.var(signal2[N]) / numpy.mean(signal2[N])
  print N,"FF:",FF[N]

# spectrogram
#------------
for N in NUCLEI:
  f,t,Sxx = sig.spectrogram(signal['STN'],fs=1000.,nperseg=100)
  pylab.pcolormesh(t,f,Sxx)
  pylab.ylabel('Frequency [Hz]')
  pylab.xlabel('Time [sec]')
  pylab.savefig(N+'_Spectrogram.pdf', bbox_inches='tight')
  #pylab.show()
'''
