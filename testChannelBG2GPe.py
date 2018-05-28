#!/apps/free/python/2.7.10/bin/python
# -*- coding: utf-8 -*-    


#import nstrand
from LGneurons2GPe import *
from modelParams import *
import nest.raster_plot
import os
import numpy as np
#import time
import sys
import matplotlib.pyplot as plt
import math
#------------------------------------------
# Creates the populations of neurons necessary to simulate a BG circuit
#------------------------------------------
def createBG_MC():
  #==========================
  # Creation of neurons
  #-------------------------
  print '\nCreating neurons\n================'
  for N in NUCLEI:
      nbSim[N] = params['nb'+N]
      createMC(N,params['nbCh'])
      for i in range(len(Pop[N])):
          nest.SetStatus(Pop[N][i],{"I_e":params['Ie'+N]})

  nbSim['CSN'] = params['nbCSN']
  createMC('CSN',params['nbCh'], fake=True, parrot=True)
  nbSim['PTN'] = params['nbPTN']
  createMC('PTN',params['nbCh'], fake=True, parrot=True)
  nbSim['CMPf'] = params['nbCMPf']
  createMC('CMPf',params['nbCh'], fake=True, parrot=params['parrotCMPf'])

  print "Number of simulated neurons:", nbSim

#------------------------------------------
# Connects the populations of a previously created multi-channel BG circuit 
#------------------------------------------
def connectBG_MC(antagInjectionSite,antag):
  #-------------------------
  # connection of populations
  #-------------------------
  print '\nConnecting neurons\n================'
  print "**",antag,"antagonist injection in",antagInjectionSite,"**"
  
  print '* MSN Inputs'
  connectMC('ex','CSN','MSN', params['cTypeCSNMSN'], inDegree= params['inDegCSNMSN'], gain=params['gainCSNMSN']) #
  connectMC('ex','PTN','MSN', params['cTypePTNMSN'], inDegree= params['inDegPTNMSN'], gain=params['gainPTNMSN']) #
  connectMC('ex','CMPf','MSN',params['cTypeCMPfMSN'],inDegree= params['inDegCMPfMSN'],gain=params['gainCMPfMSN']) #
  connectMC('in','MSN','MSN', params['cTypeMSNMSN'], inDegree= params['inDegMSNMSN'], gain=params['gainMSNMSN']) #
  connectMC('in','FSI','MSN', params['cTypeFSIMSN'], inDegree= params['inDegFSIMSN'], gain=params['gainFSIMSN']) #
  # some parameterizations from LG14 have no STN->MSN or GTA->MSN synaptic contacts
  if alpha['STN->MSN'] != 0:
    print "alpha['STN->MSN']",alpha['STN->MSN']
    connectMC('ex','STN','MSN', params['cTypeSTNMSN'], inDegree= params['inDegSTNMSN'], gain=params['gainSTNMSN']) #
  if alpha['GTA->MSN'] != 0:
    print "alpha['GTA->MSN']",alpha['GTA->MSN']
    connectMC('in','GTA','MSN', params['cTypeGTAMSN'], inDegree= params['inDegGTAMSN'], gain=params['gainGTAMSN']) #

  print '* FSI Inputs'
  connectMC('ex','CSN','FSI', params['cTypeCSNFSI'], inDegree= params['inDegCSNFSI'], gain=params['gainCSNFSI']) #
  connectMC('ex','PTN','FSI', params['cTypePTNFSI'], inDegree= params['inDegPTNFSI'], gain=params['gainPTNFSI']) #
  if alpha['STN->FSI'] != 0:
    connectMC('ex','STN','FSI', params['cTypeSTNFSI'],inDegree= params['inDegSTNFSI'],gain=params['gainSTNFSI']) #
  connectMC('in','GTA','FSI', params['cTypeGTAFSI'], inDegree= params['inDegGTAFSI'], gain=params['gainGTAFSI']) #
  connectMC('ex','CMPf','FSI',params['cTypeCMPfFSI'],inDegree= params['inDegCMPfFSI'],gain=params['gainCMPfFSI']) #
  connectMC('in','FSI','FSI', params['cTypeFSIFSI'], inDegree= params['inDegFSIFSI'], gain=params['gainFSIFSI']) #

  print '* STN Inputs'
  connectMC('ex','PTN','STN', params['cTypePTNSTN'], inDegree= params['inDegPTNSTN'],  gain=params['gainPTNSTN']) #
  connectMC('ex','CMPf','STN',params['cTypeCMPfSTN'],inDegree= params['inDegCMPfSTN'], gain=params['gainCMPfSTN']) #
  connectMC('in','GTI','STN', params['cTypeGTISTN'], inDegree= params['inDegGTISTN'],  gain=params['gainGTISTN']) #

  
  if antagInjectionSite == 'GPe':
    if   antag == 'AMPA':
      print '* GTA Inputs'
      connectMC('NMDA','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=params['gainCMPfGTA']) #
      connectMC('NMDA','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=params['gainSTNGTA']) #
      connectMC('in','MSN','GTA',   params['cTypeMSNGTA'], inDegree= params['inDegMSNGTA'], gain=params['gainMSNGTA']) #
      connectMC('in','GTA','GTA',   params['cTypeGTAGTA'], inDegree= params['inDegGTAGTA'], gain=params['gainGTAGTA']) #
      connectMC('in','GTI','GTA',   params['cTypeGTIGTA'], inDegree= params['inDegGTIGTA'], gain=params['gainGTIGTA']) #
      print '* GTI Inputs'
      connectMC('NMDA','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=params['gainCMPfGTI']) #
      connectMC('NMDA','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=params['gainSTNGTI']) #
      connectMC('in','MSN','GTI',   params['cTypeMSNGTI'], inDegree= params['inDegMSNGTI'], gain=params['gainMSNGTI']) #
      connectMC('in','GTI','GTI',   params['cTypeGTIGTI'], inDegree= params['inDegGTIGTI'], gain=params['gainGTIGTI']) #
      connectMC('in','GTA','GTI',   params['cTypeGTAGTI'], inDegree= params['inDegGTAGTI'], gain=params['gainGTAGTI']) #
      
    elif antag == 'NMDA':
      print '* GTA Inputs'
      connectMC('AMPA','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=params['gainCMPfGTA']) #
      connectMC('AMPA','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=params['gainSTNGTA']) #
      connectMC('in','MSN','GTA',   params['cTypeMSNGTA'], inDegree= params['inDegMSNGTA'], gain=params['gainMSNGTA']) #
      connectMC('in','GTA','GTA',   params['cTypeGTAGTA'], inDegree= params['inDegGTAGTA'], gain=params['gainGTAGTA']) #
      connectMC('in','GTI','GTA',   params['cTypeGTIGTA'], inDegree= params['inDegGTIGTA'], gain=params['gainGTIGTA']) #
      print '* GTI Inputs'
      connectMC('AMPA','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=params['gainCMPfGTI']) #
      connectMC('AMPA','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=params['gainSTNGTI']) #
      connectMC('in','MSN','GTI',   params['cTypeMSNGTI'], inDegree= params['inDegMSNGTI'], gain=params['gainMSNGTI']) #
      connectMC('in','GTI','GTI',   params['cTypeGTIGTI'], inDegree= params['inDegGTIGTI'], gain=params['gainGTIGTI']) #
      connectMC('in','GTA','GTI',   params['cTypeGTAGTI'], inDegree= params['inDegGTAGTI'], gain=params['gainGTAGTI']) #
    elif antag == 'AMPA+GABAA':
      print '* GTA Inputs'
      connectMC('NMDA','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=params['gainCMPfGTA']) #
      connectMC('NMDA','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=params['gainSTNGTA']) #
      print '* GTI Inputs'
      connectMC('NMDA','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=params['gainCMPfGTI']) #
      connectMC('NMDA','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=params['gainSTNGTI']) #
      
    elif antag == 'GABAA':
      print '* GTA Inputs'
      connectMC('ex','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=params['gainCMPfGTA']) #
      connectMC('ex','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=params['gainSTNGTA']) #
      print '* GTI Inputs'
      connectMC('ex','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=params['gainCMPfGTI']) #
      connectMC('ex','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=params['gainSTNGTI']) #
      
    else:
      print antagInjectionSite,": unknown antagonist experiment:",antag    
    
  else:
    print '* GTA Inputs'
    connectMC('ex','CMPf','GTA',params['cTypeCMPfGTA'],inDegree= params['inDegCMPfGTA'],gain=params['gainCMPfGTA']) #
    connectMC('ex','STN','GTA', params['cTypeSTNGTA'], inDegree= params['inDegSTNGTA'], gain=params['gainSTNGTA']) #
    connectMC('in','MSN','GTA', params['cTypeMSNGTA'], inDegree= params['inDegMSNGTA'], gain=params['gainMSNGTA']) #
    connectMC('in','GTA','GTA', params['cTypeGTAGTA'], inDegree= params['inDegGTAGTA'], gain=params['gainGTAGTA']) #
    connectMC('in','GTI','GTA', params['cTypeGTIGTA'], inDegree= params['inDegGTIGTA'], gain=params['gainGTIGTA']) #
    print '* GTI Inputs'
    connectMC('ex','CMPf','GTI',params['cTypeCMPfGTI'],inDegree= params['inDegCMPfGTI'],gain=params['gainCMPfGTI']) #
    connectMC('ex','STN','GTI', params['cTypeSTNGTI'], inDegree= params['inDegSTNGTI'], gain=params['gainSTNGTI']) #
    connectMC('in','MSN','GTI', params['cTypeMSNGTI'], inDegree= params['inDegMSNGTI'], gain=params['gainMSNGTI']) #
    connectMC('in','GTI','GTI', params['cTypeGTIGTI'], inDegree= params['inDegGTIGTI'], gain=params['gainGTIGTI']) #
    connectMC('in','GTA','GTI', params['cTypeGTAGTI'], inDegree= params['inDegGTAGTI'], gain=params['gainGTAGTI']) #

  print '* GPi Inputs'
  if antagInjectionSite =='GPi':
    if   antag == 'AMPA+NMDA+GABAA':
      pass
    elif antag == 'NMDA':
      connectMC('in','MSN','GPi',   params['cTypeMSNGPi'], inDegree= params['inDegMSNGPi'], gain=params['gainMSNGPi']) #
      connectMC('AMPA','STN','GPi', params['cTypeSTNGPi'], inDegree= params['inDegSTNGPi'], gain=params['gainSTNGPi']) #
      connectMC('in','GTI','GPi',   params['cTypeGTIGPi'], inDegree= params['inDegGTIGPi'], gain=params['gainGTIGPi']) #
      connectMC('AMPA','CMPf','GPi',params['cTypeCMPfGPi'],inDegree= params['inDegCMPfGPi'],gain=params['gainCMPfGPi']) #
    elif antag == 'NMDA+AMPA':
      connectMC('in','MSN','GPi', params['cTypeMSNGPi'],inDegree= params['inDegMSNGPi'],    gain=params['gainMSNGPi']) #
      connectMC('in','GTI','GPi', params['cTypeGTIGPi'],inDegree= params['inDegGTIGPi'],    gain=params['gainGTIGPi']) #
    elif antag == 'AMPA':
      connectMC('in','MSN','GPi',   params['cTypeMSNGPi'], inDegree= params['inDegMSNGPi'], gain=params['gainMSNGPi']) #
      connectMC('NMDA','STN','GPi', params['cTypeSTNGPi'], inDegree= params['inDegSTNGPi'], gain=params['gainSTNGPi']) #
      connectMC('in','GTI','GPi',   params['cTypeGTIGPi'], inDegree= params['inDegGTIGPi'], gain=params['gainGTIGPi']) #
      connectMC('NMDA','CMPf','GPi',params['cTypeCMPfGPi'],inDegree= params['inDegCMPfGPi'],gain=params['gainCMPfGPi']) #
    elif antag == 'GABAA':
      connectMC('ex','STN','GPi', params['cTypeSTNGPi'], inDegree= params['inDegSTNGPi'],   gain=params['gainSTNGPi']) #
      connectMC('ex','CMPf','GPi',params['cTypeCMPfGPi'],inDegree= params['inDegCMPfGPi'],  gain=params['gainCMPfGPi']) #
    else:
      print antagInjectionSite,": unknown antagonist experiment:",antag
  else:
    connectMC('in','MSN','GPi', params['cTypeMSNGPi'], inDegree= params['inDegMSNGPi'], gain=params['gainMSNGPi']) #
    connectMC('ex','STN','GPi', params['cTypeSTNGPi'], inDegree= params['inDegSTNGPi'], gain=params['gainSTNGPi']) #
    connectMC('in','GTI','GPi', params['cTypeGTIGPi'], inDegree= params['inDegGTIGPi'], gain=params['gainGTIGPi']) #
    connectMC('ex','CMPf','GPi',params['cTypeCMPfGPi'],inDegree= params['inDegCMPfGPi'],gain=params['gainCMPfGPi']) #

#------------------------------------------
# Checks that the BG model parameterization defined by the "params" dictionary can respect the electrophysiological constaints (firing rate at rest).
# If testing for a given antagonist injection experiment, specifiy the injection site in antagInjectionSite, and the type of antagonists used in antag.
# Returns [score obtained, maximal score]
# params possible keys:
# - nb{MSN,FSI,STN,GPi,GPe,CSN,PTN,CMPf} : number of simulated neurons for each population
# - Ie{GPe,GPi} : constant input current to GPe and GPi
# - G{MSN,FSI,STN,GPi,GPe} : gain to be applied on LG14 input synaptic weights for each population
#------------------------------------------

def checkAvgFR(showRasters=False,params={},antagInjectionSite='none',antag='',logFileName=''):
  nest.ResetKernel()
  dataPath='log/'
  nest.SetKernelStatus({'local_num_threads': params['nbcpu'] if ('nbcpu' in params) else 2, "data_path": dataPath})
  initNeurons()

  offsetDuration = 1000.
  # nest.SetKernelStatus({"overwrite_files":True}) # Thanks to use of timestamps, file names should now 
                                                   # be different as long as they are not created during the same second

  print '/!\ Using the following LG14 parameterization',params['LG14modelID']
  loadDictParams(params['LG14modelID'])

  # We check that all the necessary parameters have been defined. They should be in the modelParams.py file.
  # If one of them misses, we exit the program.
  necessaryParams=['nbCh','nbMSN','nbFSI','nbSTN','nbGTI','nbGTA','nbGPi','nbCSN','nbPTN','nbCMPf',
                   'IeMSN','IeFSI','IeSTN','IeGTI','IeGTA','IeGPi',
                   'gainCSNMSN','gainCSNFSI','gainPTNMSN','gainPTNFSI','gainPTNSTN',
                   'gainCMPfMSN','gainCMPfFSI','gainCMPfSTN','gainCMPfGTA','gainCMPfGTI','gainCMPfGPi',
                   'gainMSNMSN','gainMSNGTA','gainMSNGTI','gainMSNGPi','gainFSIMSN','gainFSIFSI',
                   'gainSTNMSN','gainSTNFSI','gainSTNGTA','gainSTNGTI','gainSTNGPi',
                   'gainGTAMSN','gainGTAFSI','gainGTAGTA','gainGTAGTI',
                   'gainGTISTN','gainGTIGTA','gainGTIGTI','gainGTIGPi',
                   'inDegCSNMSN','inDegPTNMSN','inDegCMPfMSN','inDegMSNMSN','inDegFSIMSN','inDegSTNMSN','inDegGTAMSN',
                   'inDegCSNFSI','inDegPTNFSI','inDegSTNFSI','inDegGTAFSI','inDegCMPfFSI','inDegFSIFSI',
                   'inDegPTNSTN','inDegCMPfSTN','inDegGTISTN',
                   'inDegCMPfGTA','inDegSTNGTA','inDegMSNGTA','inDegGTAGTA','inDegGTIGTA',
                   'inDegCMPfGTI','inDegSTNGTI','inDegMSNGTI','inDegGTIGTI','inDegGTAGTI',
                   'inDegMSNGPi','inDegSTNGPi','inDegGTIGPi','inDegCMPfGPi',]
  
  for np in necessaryParams:
    if np not in params:
      print "Missing parameter:",np 
      exit()

  #-------------------------
  # creation and connection of the neural populations
  #-------------------------
  createBG_MC()
  connectBG_MC(antagInjectionSite,antag)

  #-------------------------
  # measures
  #-------------------------
  spkDetect={} # spike detectors used to record the experiment
  expeRate={}

  antagStr = ''
  if antagInjectionSite != 'none':
    antagStr = antagInjectionSite+'_'+antag+'_'

  for N in NUCLEI:
    # 1000ms offset period for network stabilization
    spkDetect[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+N, "to_memory": False, "to_file": True, 'start':offsetDuration,'stop':offsetDuration+params['tSimu']})
    for i in range(len(Pop[N])):
      nest.Connect(Pop[N][i], spkDetect[N])

  spkDetect['CMPf'] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+'CMPf', "to_memory": False, "to_file": True, 'start':offsetDuration,'stop':offsetDuration+params['tSimu']})
  for i in range(len(Pop['CMPf'])):
    nest.Connect(Pop['CMPf'][i], spkDetect['CMPf'])

  #-------------------------
  # Simulation
  #-------------------------
  nest.Simulate(params['tSimu']+offsetDuration)

  score = 0

  text=[]
  frstr = antagInjectionSite + ', '
  s = '----- RESULTS -----'
  print s
  text.append(s+'\n')
  if antagInjectionSite == 'none':
    for N in NUCLEI:
      strTestPassed = 'NO!'
      expeRate[N] = nest.GetStatus(spkDetect[N], 'n_events')[0] / float(nbSim[N]*params['tSimu']*params['nbCh']) * 1000
      if expeRate[N] <= FRRNormal[N][1] and expeRate[N] >= FRRNormal[N][0]:
        # if the measured rate is within acceptable values
        strTestPassed = 'OK'
        score += 1
      frstr += '%f , ' %(expeRate[N])
      s = '* '+N+' - Rate: '+str(expeRate[N])+' Hz -> '+strTestPassed+' ('+str(FRRNormal[N][0])+' , '+str(FRRNormal[N][1])+')'
      print s
      text.append(s+'\n')
  else:
    for N in NUCLEI:
      expeRate[N] = nest.GetStatus(spkDetect[N], 'n_events')[0] / float(nbSim[N]*params['tSimu']*params['nbCh']) * 1000
      if N == antagInjectionSite:
        strTestPassed = 'NO!'
        if expeRate[N] <= FRRAnt[N][antag][1] and expeRate[N] >= FRRAnt[N][antag][0]:
          # if the measured rate is within acceptable values
          strTestPassed = 'OK'
          score += 1
        s = '* '+N+' with '+antag+' antagonist(s): '+str(expeRate[N])+' Hz -> '+strTestPassed+' ('+str(FRRAnt[N][antag][0])+' , '+str(FRRAnt[N][antag][1])+')'
        print s
        text.append(s+'\n')
      else:
        s = '* '+N+' - Rate: '+str(expeRate[N])+' Hz'
        print s
        text.append(s+'\n')
      frstr += '%f , ' %(expeRate[N])

  s = '-------------------'
  print s
  text.append(s+'\n')

  frstr+='\n'
  firingRatesFile=open(dataPath+'firingRates.csv','a')
  firingRatesFile.writelines(frstr)
  firingRatesFile.close()

  #print "************************************** file writing",text
  #res = open(dataPath+'OutSummary_'+logFileName+'.txt','a')
  res = open(dataPath+'OutSummary.txt','a')
  res.writelines(text)
  res.close()

  #-------------------------
  # Displays
  #-------------------------
  if showRasters and interactive:
    displayStr = ' ('+antagStr[:-1]+')' if (antagInjectionSite != 'none') else ''
    for N in NUCLEI:
      #nest.raster_plot.from_device(spkDetect[N],hist=True,title=N+displayStr)
      nest.raster_plot.from_device(spkDetect[N],hist=False,title=N+displayStr)

    nest.raster_plot.from_device(spkDetect['CMPf'],hist=False,title='CMPf'+displayStr)

    nest.raster_plot.show()

  return score, 6 if antagInjectionSite == 'none' else 1

# -----------------------------------------------------------------------------
# This function verify if their is pauses in the GPe and if the caracteristiques
# of theses pauses are relevant with the data of the elias paper 2007 
# It is run after CheckAVGFR because it uses the gdf files of the simulation.
# -----------------------------------------------------------------------------

#---------------------------- begining getSpikes ------------------------------
# return an ordered dictionnary of the spikes occurences by neuron and in the time
def getSpikes(Directory, Nuclei):
    spikesDict = {}
    spikesList = []
    gdfList = os.listdir(Directory + '/NoeArchGdf')
    
    for f in gdfList:
        if f.find(Nuclei) != -1 and f[-4:] == ".gdf" :
            spikeData = open(Directory +'/NoeArchGdf/' + f)
            for line in spikeData: # take the spike and put it in neuronRecording
                spk = line.split('\t')
                spk.pop()
                spikesList.append(float(spk[1]))
                if spk[0] in spikesDict:
                    spikesDict[spk[0]].append(float(spk[1]))
                else:
                    spikesDict[spk[0]] = [float(spk[1])]
        
    for neuron in spikesDict:
        spikesDict[neuron] = sorted(spikesDict[neuron])
    
    return spikesDict, spikesList
#---------------------------- end getSpikes -----------------------------------
    
#--------------------------- begining getISIs ---------------------------------
# return ISIs ordered by neuron in a dictionnary
def getISIs(spikesDict):
    ISIsDict = {}
    for neuron in spikesDict:
        ISIsDict[neuron] = []
        for i in range(len(spikesDict[neuron]) - 1):
            ISIsDict[neuron].append(round(spikesDict[neuron][i+1] - spikesDict[neuron][i], 1))
    ISIsList = []
    for neuron in ISIsDict:
        for isi in ISIsDict[neuron]:
            ISIsList.append(isi)       
    return ISIsDict, ISIsList
#----------------------------- end getISIs ------------------------------------ 
    
#--------------------------- begining rasterPlot ------------------------------
# plot rasters figures in the directory /raster
def rasterPlot(spikesDict, Nuclei, Directory):
    rasterList = []
    
    if not os.path.exists(Directory + '/rasterPlot'):
        os.makedirs(Directory + '/rasterPlot')

    for neuron in spikesDict:
        rasterList.append(spikesDict[neuron])  
    plt.figure(figsize=(40,15))
    plt.eventplot(rasterList, linelengths = 0.8, linewidths = 0.6)
    plt.title('Spike raster plot ' + Nuclei)
    plt.grid()
    plt.savefig(Directory + '/rasterPlot/' + 'RasterPlot_' + Nuclei + '.png')
#----------------------------- end rasterPlot ---------------------------------
    
#--------------------------- begining BarPlot ---------------------------------
# plot the nuclei histogram of ISIs
def activityHistPlot(spikesList, Nuclei, Directory):
    
    if not os.path.exists(Directory + '/activityHistPlot'):
        os.makedirs(Directory + '/activityHistPlot')

    plt.figure(figsize=(40,5))
    plt.hist(spikesList, bins=200, normed=0.5)
    plt.title('Histogram of the activity' + Nuclei)
    plt.grid()
    plt.savefig(Directory + '/activityHistPlot/'+ 'activityHistPlot_' + Nuclei + '.png')
#----------------------------- end BarPlot ------------------------------------
    
#--------------------------- begining BarPlot ---------------------------------
# plot the nuclei histogram of ISIs
def HistPlot(ISIsList, Nuclei, Directory):
    
    if not os.path.exists(Directory + '/histPlot'):
        os.makedirs(Directory + '/histPlot')
        
    plt.figure()
    plt.hist(ISIsList, bins=20, normed=0.5)
    plt.title('Histogram ' + Nuclei)
    plt.grid()
    plt.savefig(Directory + '/histPlot/'+ 'HistPlot_' + Nuclei + '.png')
#----------------------------- end BarPlot ------------------------------------
    
#--------------------------- begining poisson ---------------------------------
# compute the poissonian probability that n or less spike occure during T ms
def poisson(n, r, T): # Tsum of 2 isi or 3 ? n = 2
    P = 0
    for i in range(n):
        P += math.pow(r*T, i)/ math.factorial(i)

    return P*math.exp(-r*T)
#----------------------------- end poisson ------------------------------------

#----------------------- begining Pause Analysis ------------------------------
def PauseAnalysis(ISIsDict,ISIsList): # Tsum of 2 isi or 3 ? n = 2
    simuSpecs = {'meanISI': np.mean(ISIsList),}
    
    r = 1/float(simuSpecs['meanISI'])
    pausesDict = {}
    pausesList = []
    coreIList = []
    
    isiThreshold = 0

    if max(ISIsList) >= 250:
        isiThreshold = 250
    elif max(ISIsList) >= 200:
        isiThreshold = 200
    elif max(ISIsList) >= 150:
        isiThreshold = 150
    elif max(ISIsList) >= 100:
        isiThreshold = 100
    elif max(ISIsList) >= 80:
        isiThreshold = 80
    elif max(ISIsList) >= 60:
        isiThreshold = 60
    elif max(ISIsList) >= 40:
        isiThreshold = 40
    else:
        isiThreshold = 20
          
    for neuron in ISIsDict:
        skip = False
        for i in range(1,len(ISIsDict[neuron])-1):
            if ISIsDict[neuron][i] >= isiThreshold and not skip :
                coreI = ISIsDict[neuron][i]
                pause = coreI
                s = -math.log10(poisson(1, r, coreI))
                s2 = -math.log10(poisson(2, r, coreI+ISIsDict[neuron][i-1]))
                s3 = -math.log10(poisson(2, r, coreI+ISIsDict[neuron][i+1]))
                if s2 > s and s2 >= s3:
                    s = s2
                    pause += ISIsDict[neuron][i-1]
                elif s3 > s:
                    s = s3
                    pause += ISIsDict[neuron][i+1]
                    skip = True
        
                if neuron in pausesDict:
                    pausesDict[neuron].append(pause)
                    pausesList.append(pause)
                    coreIList.append(coreI)
                else:
                    pausesDict[neuron] = [pause]
                    pausesList.append(pause)
                    coreIList.append(coreI)
            else:
                skip = False
        
        pausersFRRList = []
        correctedFRRList = []
        for neuron in pausesDict:
            pausersFRRList.append((len(ISIsDict[neuron])+1)*1000/float(params['tSimu']))
            pausesLength = sum(pausesDict[neuron])
            correctedFRRList.append((len(ISIsDict[neuron])-len(pausesDict[neuron])+1)*1000/(float(params['tSimu'])-pausesLength))
            
            
    
    simuSpecs['isiThreshold'] = isiThreshold
    simuSpecs['percentagePausers'] = len(pausesDict)/float(len(ISIsDict))*100
    simuSpecs['nbPausersNeurons'] = len(pausesDict)
    simuSpecs['meanPausesDuration'] = round(np.mean(pausesList),2)
    simuSpecs['meanCoreI'] = round(np.mean(coreIList),2)
    simuSpecs['nbPausesPerMin'] = round(len(pausesList)/float(len(pausesDict)*params['tSimu'])*60000,2)
    simuSpecs['nbPauses'] = len(pausesList)
    simuSpecs['meanISI'] = round(np.mean(ISIsList),2)
    simuSpecs['pausersFRR'] = round(np.mean(pausersFRRList),2)
    simuSpecs['minPausersFRR'] = round(min(pausersFRRList),2)
    simuSpecs['correctedPausersFRR'] = round(np.mean(correctedFRRList),2)

    return simuSpecs
#-------------------------- end Pause Analysis --------------------------------
    
#------------------------- begining gdf exploitation --------------------------
# call the function and plot results
def gdfExploitation(Directory):
    pausesDATA = {'percentagePausers':  [40. ,    100.,   75.],        # percentage of pauser neurons in GPe [low value, High value, perfect value]
                  'shortPercentageISI': [0 ,     0.70,    0.2],         # percentage of Interspike intervals inferior to 2 ms
                  'meanPausesDuration': [450. ,  730.,   620.],     # change to [0.45, 0.73, 0.62] are the  extreme recorded values if it is too selective
                  'nbPausesPerMin':     [8. ,     23.,    13.],            # change to [8, 23, 13] are the  extreme recorded values if it is too selective
                  'meanIPI':            [2.63 ,  8.74,   6.19],     # InterPauses Inteval | [2.63, 8.74, 6.19]are the  extreme recorded values if it is too selective
                  'pausersFRR':         [37.48 , 71.25, 54.37],  # change to [21.47, 76.04, 54.13] which are the  extreme recorded values if it is too selective
                  'correctedPausersFRR':[44.04 , 81.00, 62.52],  # change to [22.60, 86.63, 62.52] which are the  extreme recorded values if it is too selective
                  'nonPausersFRR':      [37.10 , 75.75, 56.43],} # change to [31.37, 91.70, 56.43] which are the  extreme recorded values if it is too selective
    
    for N in NUCLEI:
        a = getSpikes(Directory, N)
        spikesDict = a[0]
        activityHistPlot(a[1], N, Directory)
        rasterPlot(spikesDict, N, Directory)
        
        if N == 'GTA' or N == 'GTI':
            
            simuSpecs = PauseAnalysis(getISIs(spikesDict)[0], getISIs(spikesDict)[1])
            
            text = "\n################# Pause Results " + N + " #################"
            text += "\n ISI threshold       = " + str(simuSpecs['isiThreshold']) + " ms    | 250 ms"
            text += "\n Mean coreI duration = " + str(simuSpecs['meanCoreI']) + " ms | [200 - 600]"
            text += "\n Mean pause duration = " + str(simuSpecs['meanPausesDuration']) + " ms | 620 ms"
            text += "\n Mean ISI            = " + str(simuSpecs['meanISI']) + "  ms | 15 ms"
            text += "\n total Pauses Nb     = " + str(simuSpecs['nbPauses']) 
            text += "\n pause/min/neuron    = " + str(simuSpecs['nbPausesPerMin']) + "    | [13 - 24]"
            text += "\n Pauser neurons Nb   = " + str(simuSpecs['nbPausersNeurons'])
            text += "\n % Pauser neurons    = " + str(simuSpecs['percentagePausers'])  + "     | [60 - 100]\n"
            text += "\n pausersFRR          = " + str(simuSpecs['pausersFRR']) + "    | [37 - 54]"
            text += "\n corr pausers FRR    = " + str(simuSpecs['correctedPausersFRR']) + "     | [44 - 62]"
            text += "\n Min pausers FRR     = " + str(simuSpecs['minPausersFRR']) + "      | [30 - 54] \n"
            text += "\n#####################################################\n"
            
            res = open(Directory+'/log/OutSummary.txt','a')
            res.writelines(text)
            res.close()
            print text

#---------------------------- end gdf exploitation ----------------------------


#-----------------------------------------------------------------------
# PActiveCNS/PTN : proportion of "active" neurons in the CSN/PTN populations (in [0.,1.])
#
#-----------------------------------------------------------------------
def checkGurneyTest(showRasters=False,params={},CSNFR=[2.,10.], PActiveCSN=1., PTNFR=[15.,35], PActivePTN=1., antagInjectionSite='none',antag=''):
  nest.ResetKernel()
  dataPath='log/'
  nest.SetKernelStatus({'local_num_threads': params['nbcpu'] if ('nbcpu' in params) else 2, "data_path": dataPath})
  initNeurons()

  offsetDuration = 200.
  simDuration = 800. # ms
  loadLG14params(params['LG14modelID'])

  # We check that all the necessary parameters have been defined. They should be in the modelParams.py file.
  # If one of them misses, we exit the program.
  necessaryParams=['nbCh','nbMSN','nbFSI','nbSTN','nbGPe','nbGPi','nbCSN','nbPTN','nbCMPf','IeGPe','IeGPi','GMSN','GFSI','GSTN','GGPe','GGPi','inDegCSNMSN','inDegPTNMSN','inDegCMPfMSN','inDegMSNMSN','inDegFSIMSN','inDegSTNMSN','inDegGPeMSN','inDegCSNFSI','inDegPTNFSI','inDegSTNFSI','inDegGPeFSI','inDegCMPfFSI','inDegFSIFSI','inDegPTNSTN','inDegCMPfSTN','inDegGPeSTN','inDegCMPfGPe','inDegSTNGPe','inDegMSNGPe','inDegGPeGPe','inDegMSNGPi','inDegSTNGPi','inDegGPeGPi','inDegCMPfGPi',]
  for nepa in necessaryParams:
    if nepa not in params:
      print "Missing parameter:",nepa
      exit()

  nbRecord=2 # number of channels whose activity will be recorded
  if params['nbCh']<2:
    print 'need at least 2 channels to perform Gurney test'
    exit()
  elif params['nbCh']>2:
    nbRecord = 3 # if we have more than 2 channels, we will also record one of the neutral channels

  #-------------------------
  # creation and connection of the neural populations
  #-------------------------
  createBG_MC()
  connectBG_MC(antagInjectionSite,antag)

  #-------------------------
  # prepare the firing rates of the inputs for the 5 steps of the experiment
  #-------------------------  
  gCSN = CSNFR[1]-CSNFR[0]
  gPTN = PTNFR[1]-PTNFR[0]
  activityLevels = np.array([[0,0.4,0.4,0.6,0.4], [0.,0.,0.6,0.6,0.6]]) 

  CSNrate= gCSN * activityLevels + np.ones((5)) * CSNFR[0]
  PTNrate= gPTN * activityLevels + np.ones((5)) * PTNFR[0]

  #-------------------------
  # and prepare the lists of neurons that will be affected by these activity changes
  #-------------------------
  ActPop = {'CSN':[(),()],'PTN':[(),()]}
  if 'Fake' in globals():
    if 'CSN' in Fake:
      if PActiveCSN==1.:
       ActPop['CSN']=Fake['CSN']
      else:
        for i in range(2):
          ActPop['CSN'][i] = tuple(rnd.choice(a=np.array(Fake['CSN'][i]),size=int(nbSim['CSN']*PActiveCSN),replace=False))
    else:
      if PActiveCSN==1.:
       ActPop['CSN']=Pop['CSN']
      else:
        for i in range(2):
          ActPop['CSN'][i] = tuple(rnd.choice(a=np.array(Pop['CSN'][i]),size=int(nbSim['CSN']*PActiveCSN),replace=False))
    if 'PTN' in Fake:
      if PActivePTN==1.:
        ActPop['PTN']=Fake['PTN']
      else:
        for i in range(2):
          ActPop['PTN'][i] = tuple(rnd.choice(a=np.array(Fake['PTN'][i]),size=int(nbSim['PTN']*PActivePTN),replace=False))
    else:
      if PActivePTN==1.:
        ActPop['PTN']=Pop['PTN']
      else:
        for i in range(2):
          ActPop['PTN'][i] = tuple(rnd.choice(a=np.array(Pop['PTN'][i]),size=int(nbSim['PTN']*PActivePTN),replace=False))
  else:
    if PActiveCSN==1.:
     ActPop['CSN']=Pop['CSN']
    else:
      for i in range(2):
        ActPop['CSN'][i] = tuple(rnd.choice(a=np.array(Pop['CSN'][i]),size=int(nbSim['CSN']*PActiveCSN),replace=False))
    if PActivePTN==1.:
      ActPop['PTN']=Pop['PTN']
    else:
      for i in range(2):
        ActPop['PTN'][i] = tuple(rnd.choice(a=np.array(Pop['PTN'][i]),size=int(nbSim['PTN']*PActivePTN),replace=False))

  #-------------------------
  # log-related variales
  #-------------------------
  score = 0
  expeRate={}
  for N in NUCLEI:
    expeRate[N]=-1. * np.ones((nbRecord,5))

  inspector = {}
  for N in NUCLEI:
    inspector[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": N, "to_file": False})
    for i in range(nbRecord):
      nest.Connect(Pop[N][i],inspector[N])

  #-------------------------
  # write header in firingRate summary file
  #-------------------------
  frstr = 'step , '
  for i in range(nbRecord):
    for N in NUCLEI:
      frstr += N+' ('+str(i)+') , '
  frstr+='\n'
  firingRatesFile=open(dataPath+'firingRates.csv','w')
  firingRatesFile.writelines(frstr)

  #----------------------------------
  # Loop over the 5 steps of the test
  #----------------------------------
  for timeStep in range(5):
    #-------------------------
    # measures                                                                                                                                                  
    #-------------------------
    spkDetect=[{},{},{}] # list of spike detector dictionaries used to record the experiment in the first 3 channels

    antagStr = ''
    if antagInjectionSite != 'none':
      antagStr = antagInjectionSite+'_'+antag+'_'

    for i in range(nbRecord):
      for N in NUCLEI:
        spkDetect[i][N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": str(timeStep)+'_'+antagStr+N, "to_file": True, 'start':offsetDuration + timeStep*(offsetDuration+simDuration),'stop':(timeStep+1)*(offsetDuration+simDuration)})
        nest.Connect(Pop[N][i], spkDetect[i][N])

    frstr = str(timeStep) + ', '

    #-------------------------
    # Simulation
    #-------------------------
    print '====== Step',timeStep,'======'
    print 'Channel 0:',CSNrate[0,timeStep],PTNrate[0,timeStep]
    print 'Channel 1:',CSNrate[1,timeStep],PTNrate[1,timeStep]

    nest.SetStatus(ActPop['CSN'][0],{'rate':CSNrate[0,timeStep]})
    nest.SetStatus(ActPop['CSN'][1],{'rate':CSNrate[1,timeStep]})
    nest.SetStatus(ActPop['PTN'][0],{'rate':PTNrate[0,timeStep]})
    nest.SetStatus(ActPop['PTN'][1],{'rate':PTNrate[1,timeStep]})

    nest.Simulate(simDuration+offsetDuration)

    for i in range(nbRecord):
      print '------ Channel',i,'-------'
      for N in NUCLEI:
        #strTestPassed = 'NO!'
        expeRate[N][i,timeStep] = nest.GetStatus(spkDetect[i][N], 'n_events')[0] / float(nbSim[N]*simDuration) * 1000
        print 't('+str(timeStep)+')',N,':',expeRate[N][i,timeStep],'Hz'
        frstr += '%f , ' %(expeRate[N][i,timeStep])

    strTestPassed = 'YES!'
    if timeStep == 0:
      for i in range(params['nbCh']):
        if expeRate['GPi'][0,timeStep]<FRRNormal['GPi'][0]:
          strTestPassed = 'NO!'
      meanRestGPi = expeRate['GPi'][:,timeStep].mean()
    elif timeStep == 1:
      if expeRate['GPi'][0,timeStep] > meanRestGPi*0.9:
        strTestPassed = 'NO!'
    elif timeStep == 2:
      if expeRate['GPi'][1,timeStep] > meanRestGPi*0.9 or expeRate['GPi'][0,timeStep] < expeRate['GPi'][1,timeStep]:
        strTestPassed = 'NO!'
    elif timeStep == 3:
      if expeRate['GPi'][0,timeStep] > meanRestGPi*0.9 or expeRate['GPi'][1,timeStep] > meanRestGPi*0.9 :
        strTestPassed = 'NO!'
    elif timeStep == 4:
      if expeRate['GPi'][1,timeStep] > meanRestGPi*0.9 or expeRate['GPi'][0,timeStep] < expeRate['GPi'][1,timeStep]:
        strTestPassed = 'NO!'

    if strTestPassed == 'YES!':
      score +=1

    print '------ Result ------'
    print expeRate['GPi'][0,timeStep],'Hz',expeRate['GPi'][1,timeStep],'Hz',strTestPassed

    # write measured firing rates in csv file
    frstr+='\n'
    firingRatesFile.writelines(frstr)

    #-------------------------
    # Displays
    #-------------------------
    '''
    if showRasters and interactive:
      for i in range(nbRecord):
        displayStr = ' Channel '+str(i)
        displayStr+=' ('+antagStr[:-1]+')' if (antagInjectionSite != 'none') else ''
        #for N in NUCLEI:
        for N in ['MSN','STN']:
          #nest.raster_plot.from_device(spkDetect[i][N],hist=True,title=N+displayStr)
          nest.raster_plot.from_device(spkDetect[i][N],hist=False,title=N+displayStr)
      nest.raster_plot.show()
    '''

  if showRasters and interactive:
    for N in NUCLEI:
      nest.raster_plot.from_device(inspector[N],hist=True,title=N)
    nest.raster_plot.show()

  firingRatesFile.close()

  return score,5

#-----------------------------------------------------------------------
# PActiveCNS/PTN : proportion of "active" neurons in the CSN/PTN populations (in [0.,1.])
#
#-----------------------------------------------------------------------
def checkGeorgopoulosTest(showRasters=False,params={},CSNFR=[2.,10.], PActiveCSN=1., PTNFR=[15.,35], PActivePTN=1., antagInjectionSite='none',antag=''):
  nest.ResetKernel()
  dataPath='log/'
  nest.SetKernelStatus({'local_num_threads': params['nbcpu'] if ('nbcpu' in params) else 2, "data_path": dataPath})
  initNeurons()

  offsetDuration = 500.
  simDuration = 1000. # ms
  loadLG14params(params['LG14modelID'])

  # We check that all the necessary parameters have been defined. They should be in the modelParams.py file.
  # If one of them misses, we exit the program.
  necessaryParams=['nbCh','nbMSN','nbFSI','nbSTN','nbGPe','nbGPi','nbCSN','nbPTN','nbCMPf','IeGPe','IeGPi','GMSN','GFSI','GSTN','GGPe','GGPi','inDegCSNMSN','inDegPTNMSN','inDegCMPfMSN','inDegMSNMSN','inDegFSIMSN','inDegSTNMSN','inDegGPeMSN','inDegCSNFSI','inDegPTNFSI','inDegSTNFSI','inDegGPeFSI','inDegCMPfFSI','inDegFSIFSI','inDegPTNSTN','inDegCMPfSTN','inDegGPeSTN','inDegCMPfGPe','inDegSTNGPe','inDegMSNGPe','inDegGPeGPe','inDegMSNGPi','inDegSTNGPi','inDegGPeGPi','inDegCMPfGPi',]
  for nepa in necessaryParams:
    if nepa not in params:
      print "Missing parameter:",nepa
      exit()

  #-------------------------
  # creation and connection of the neural populations
  #-------------------------
  createBG_MC()
  connectBG_MC(antagInjectionSite,antag)

  #-------------------------
  # prepare the firing rates of the inputs
  #-------------------------
  gCSN = CSNFR[1]-CSNFR[0]
  gPTN = PTNFR[1]-PTNFR[0]
  activityLevels = np.ones((params['nbCh']))
  for i in range(params['nbCh']):
    activityLevels[i] = 2 * np.pi / params['nbCh'] * i
  activityLevels = (np.cos(activityLevels)+1)/2.

  CSNrate= gCSN * activityLevels + np.ones((params['nbCh'])) * CSNFR[0]
  PTNrate= gPTN * activityLevels + np.ones((params['nbCh'])) * PTNFR[0]

  #-------------------------
  # and prepare the lists of neurons that will be affected by these activity changes
  #-------------------------
  ActPop = {'CSN':[() for i in range(params['nbCh'])],'PTN':[() for i in range(params['nbCh'])]}
  if 'Fake' in globals():
    if 'CSN' in Fake:
      if PActiveCSN==1.:
       ActPop['CSN']=Fake['CSN']
      else:
        for i in range(params['nbCh']):
          ActPop['CSN'][i] = tuple(rnd.choice(a=np.array(Fake['CSN'][i]),size=int(nbSim['CSN']*PActiveCSN),replace=False))
    else:
      if PActiveCSN==1.:
       ActPop['CSN']=Pop['CSN']
      else:
        for i in range(params['nbCh']):
          ActPop['CSN'][i] = tuple(rnd.choice(a=np.array(Pop['CSN'][i]),size=int(nbSim['CSN']*PActiveCSN),replace=False))
    if 'PTN' in Fake:
      if PActivePTN==1.:
        ActPop['PTN']=Fake['PTN']
      else:
        for i in range(params['nbCh']):
          ActPop['PTN'][i] = tuple(rnd.choice(a=np.array(Fake['PTN'][i]),size=int(nbSim['PTN']*PActivePTN),replace=False))
    else:
      if PActivePTN==1.:
        ActPop['PTN']=Pop['PTN']
      else:
        for i in range(params['nbCh']):
          ActPop['PTN'][i] = tuple(rnd.choice(a=np.array(Pop['PTN'][i]),size=int(nbSim['PTN']*PActivePTN),replace=False))
  else:
    if PActiveCSN==1.:
     ActPop['CSN']=Pop['CSN']
    else:
      for i in range(params['nbCh']):
        ActPop['CSN'][i] = tuple(rnd.choice(a=np.array(Pop['CSN'][i]),size=int(nbSim['CSN']*PActiveCSN),replace=False))
    if PActivePTN==1.:
      ActPop['PTN']=Pop['PTN']
    else:
      for i in range(params['nbCh']):
        ActPop['PTN'][i] = tuple(rnd.choice(a=np.array(Pop['PTN'][i]),size=int(nbSim['PTN']*PActivePTN),replace=False))

  #-------------------------
  # log-related variables
  #-------------------------
  GPiRestRate= -1*np.ones((params['nbCh']))
  expeRate={}
  for N in NUCLEI:
    expeRate[N]=-1. * np.ones((params['nbCh']))

  inspector = {}
  for N in NUCLEI:
    inspector[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": N, "to_file": False})
    for i in range(params['nbCh']):
      nest.Connect(Pop[N][i],inspector[N])

  #-------------------------
  # write header in firingRate summary file
  #-------------------------
  frstr = 'channel , '
  for N in NUCLEI:
    frstr += N+', '
  frstr+='\n'
  firingRatesFile=open(dataPath+'firingRates.csv','w')
  firingRatesFile.writelines(frstr)

  #-------------------------
  # measures
  #-------------------------
  spkDetect=[{} for i in range(params['nbCh'])] # list of spike detector dictionaries used to record the experiment in all the channels

  antagStr = ''
  if antagInjectionSite != 'none':
    antagStr = antagInjectionSite+'_'+antag+'_'

  for i in range(params['nbCh']):
    for N in NUCLEI:
      spkDetect[i][N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+N, "to_file": True, 'start':2*offsetDuration+simDuration,'stop':2*(offsetDuration+simDuration)})
      nest.Connect(Pop[N][i], spkDetect[i][N])

  GPiRestSpkDetect = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+'GPiRest', "to_file": True, 'start':offsetDuration,'stop':offsetDuration+simDuration})
  for i in range(params['nbCh']):
      nest.Connect(Pop['GPi'][i], GPiRestSpkDetect)

  #-------------------------
  # Simulation
  #-------------------------

  # step 1 : stimulation without inputs, to calibrate the GPi activity at rest :
  nest.Simulate(simDuration+offsetDuration)

  print '------ Rest Period ------'  
  frstr = 'rest, , , , ,' # only GPi is recorded at rest, and on all channels
  GPiRestRate = nest.GetStatus(GPiRestSpkDetect, 'n_events')[0] / float(nbSim[N]*simDuration*params['nbCh']) * 1000
  print "GPi rate at rest:",GPiRestRate;"Hz"
  frstr += '%f \n' %GPiRestRate
  firingRatesFile.writelines(frstr)

  for i in range(params['nbCh']):
    nest.SetStatus(ActPop['CSN'][i],{'rate':CSNrate[i]})
    nest.SetStatus(ActPop['PTN'][i],{'rate':PTNrate[i]})

  nest.Simulate(simDuration+offsetDuration)

  for i in range(params['nbCh']):
    print '------ Channel',i,'------'
    frstr = str(i)+', '
    for N in NUCLEI:
      expeRate[N][i] = nest.GetStatus(spkDetect[i][N], 'n_events')[0] / float(nbSim[N]*simDuration) * 1000
      print N,':',expeRate[N][i],'Hz'
      frstr += '%f , ' %(expeRate[N][i])
    frstr += '\n'

    firingRatesFile.writelines(frstr)

  firingRatesFile.close()

  #-------------------------
  # Displays
  #-------------------------
  if showRasters and interactive:
    for N in NUCLEI:
      nest.raster_plot.from_device(inspector[N],hist=True,title=N)
    nest.raster_plot.show()

    pylab.plot(expeRate['GPi'])
    pylab.show()

  contrast = 2. / CSNrate * GPiRestRate / expeRate['GPi']

  return contrast

#-----------------------------------------------------------------------
  
def main():
    
  Directory = os.getcwd()
  
  if len(sys.argv) >= 2:
    print "Command Line Parameters"
    paramKeys = ['LG14modelID',
                 'nbMSN',
                 'nbFSI',
                 'nbSTN',
                 'nbGTA',
                 'nbGTI',
                 'nbGPi',
                 'nbCSN',
                 'nbPTN',
                 'nbCMPf',
                 'GMSN',
                 'GFSI',
                 'GSTN',
                 'GGTA',
                 'GGTI',
                 'GGPi', 
                 'IeGTA',
                 'IeGTI',
                 'IeGPi',
                 'inDegCSNMSN',
                 'inDegPTNMSN',
                 'inDegCMPfMSN',
                 'inDegFSIMSN',
                 'inDegMSNMSN', 
                 'inDegCSNFSI',
                 'inDegPTNFSI',
                 'inDegSTNFSI',
                 'inDegGTAFSI',
		         'inDegGTAMSN',
                 'inDegCMPfFSI',
                 'inDegFSIFSI',
                 'inDegPTNSTN',
                 'inDegCMPfSTN',
                 'inDegGTISTN',
                 'inDegCMPfGTA',
                 'inDegCMPfGTI',
                 'inDegSTNGTA',
                 'inDegSTNGTI',
                 'inDegMSNGTA',
                 'inDegMSNGTI',
                 'inDegGTAGTA',
                 'inDegGTIGTA',
                 'inDegGTIGTI',
                 'inDegMSNGPi',
                 'inDegSTNGPi',
                 'inDegGTIGPi',
                 'inDegCMPfGPi',]
    
    if len(sys.argv) == len(paramKeys)+1:
      print "Using command line parameters"
      print sys.argv
      i = 0
      for k in paramKeys:
        i+=1
        params[k] = float(sys.argv[i])
    else :
      print "Incorrect number of parameters:",len(sys.argv),"-",len(paramKeys),"expected"

  nest.set_verbosity("M_WARNING")
  
  score = np.zeros((2)) 
  score += checkAvgFR(params=params,antagInjectionSite='none',antag='',showRasters=True)
  
  print "******************"
  print " Score FRR:",score[0],'/',score[1]
  print "******************"
  
  os.system('mkdir NoeArchGdf')  # save the .gdf files before antagonist desaster 
  os.system('cp log/MSN* log/STN* log/GTI* log/GTA* log/GPi* log/FSI* NoeArchGdf/ ')
  os.system('rm log/MSN* log/STN* log/GTI* log/GTA* log/FSI* log/CMPf*')
  gdfExploitation(Directory)
  
  if score[0] < score[1]:
    print("Activities at rest do not match: skipping deactivation tests")
#  else:
#      for a in ['AMPA','AMPA+GABAA','NMDA','GABAA']:
#        score += checkAvgFR(params=params,antagInjectionSite='GPe',antag=a)
#    
#      for a in ['AMPA+NMDA+GABAA','AMPA','NMDA+AMPA','NMDA','GABAA']:
#        score += checkAvgFR(params=params,antagInjectionSite='GPi',antag=a)
#  os.system('rm log/G*')

  #-------------------------
  print "******************"
  print "* Score:",score[0],'/',score[1]
  print "******************"

  #-------------------------
  # log the results in a file
  #-------------------------
  #res = open('OutSummary_'+timeStr+'.txt','a')
  res = open('OutSummary.txt','a')
  for k,v in params.iteritems():
    res.writelines(k+' , '+str(v)+'\n')
  res.writelines("Score: "+str(score[0])+' , '+str(score[1]))
  res.close()

  res = open('score.txt','w')
  res.writelines(str(score[0])+'\n')
  res.close()

#---------------------------
if __name__ == '__main__':
  main()
