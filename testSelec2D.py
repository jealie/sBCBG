#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## testGPR01.py
##
## This script implements the selection test from Gurney, Prescott and Redrave, 2001b

from iniBG import *

import os
import glob
import shutil

#-----------------------------------------------------------------------
# PActiveCNS/PTN : proportion of "active" neurons in the CSN/PTN populations (in [0.,1.])
#
#-----------------------------------------------------------------------
def checkSelec(params={},CSNFR=[2.,10.], PActiveCSN=1., PTNFR=[15.,35], PActivePTN=1., antagInjectionSite='none',antag='', CMPfFR=[4.,4.], PActiveCMPf=1., transientCMPf=1., simDuration=1000., step=0.1, sum_constraint = None, start=0, end=1):

  nest.ResetNetwork()
  initNeurons()

  dataPath='log/'
  nest.SetKernelStatus({"overwrite_files":True}) # when we redo the simulation, we erase the previous traces

  offsetDuration = 1000. # preliminary period (not recorded)
  if 'simDuration' not in params.keys():
    simDuration = 1000. # step duration period
  else:
    simDuration = params['simDuration']
  #simDuration = 20000.; offsetDuration=2000. # 20 s traces
  #simDuration = 10000.; offsetDuration=2000. # 10 s traces
  #simDuration = 5000.; offsetDuration=1000. # 5 s traces
  #simDuration = 100.; offsetDuration = 10.; # quick (for test)

  nbRecord=3 # number of channels whose activity will be recorded
  if params['nbCh'] != 3:
    print 'need exactly 3 channels to perform the 2D selection test'
    exit()

  #-------------------------
  # prepare the firing rates of the inputs for the 3 steps of the experiment
  #-------------------------  
  gCSN = CSNFR[1]-CSNFR[0]
  gPTN = PTNFR[1]-PTNFR[0]

  if sum_constraint == None:
    # no constraint
    activityLevels = np.array([[x,y,0.] for x in np.arange(start,end+step,step) for y in np.arange(start,end+step,step)]).transpose()
  else:
    activityLevels = np.array([[x,y,0.] if np.abs(x+y-sum_constraint)<step/10 else [-1,-1,-1] for x in np.arange(start,end+step,step) for y in np.arange(start,end+step,step)])
    activityLevels = activityLevels[np.where(activityLevels[:,0] != -1), :][0, :, :].transpose()

  nsteps = len(activityLevels[0])
  CSNrate = gCSN * activityLevels + np.ones((nsteps)) * CSNFR[0]
  PTNrate = gPTN * activityLevels + np.ones((nsteps)) * PTNFR[0]

  #-------------------------
  # and prepare the lists of neurons that will be affected by these activity changes
  #-------------------------
  ActPop = {'CSN':[(),()],'PTN':[(),()],'CMPf':[(),(),()]}
  def activate_pop(N, PActive, nbCh=2):
    src = Pop[n]
    if 'Fake' in globals():
      if N in Fake:
        src = Fake[N]
    if PActive==1.:
     ActPop[N]=src
    else:
      for i in range(nbCh):
        ActPop[N][i] = tuple(rnd.choice(a=np.array(src[i]),size=int(nbSim[N]*PActive),replace=False))
  activate_pop('CSN', PActiveCSN)
  activate_pop('PTN', PActivePTN)
  activate_pop('CMPf', PActiveCMPf, nbCh=params['nbCh'])

  #-------------------------
  # log-related variales
  #-------------------------
  score = 0
  expeRate={}
  for N in NUCLEI:
    expeRate[N]=-1. * np.ones((nbRecord,3))

  antagStr = ''
  if antagInjectionSite != 'none':
    antagStr = antagInjectionSite+'_'+antag+'_'

  last_n_events = {} # used to compute firing rates
  inspector = {}
  for N in ['CSN','PTN','CMPf']+NUCLEI:
    inspector[N] = [None for i in range(nbRecord)] # channel-specific recording
    last_n_events[N] = [0. for i in range(nbRecord)] # channel-specific firing rate storing
    for i in range(nbRecord):
      inspector[N][i] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": N+'_'+str(i), "to_file": True, 'start':0. ,'stop':(offsetDuration+simDuration)*nsteps})
      nest.Connect(Pop[N][i],inspector[N][i])

  

  #-------------------------
  # firingRate summary
  #-------------------------
  frstr = 'Ch1 , Ch2, Tstart, Tend, '
  #for N in ['CSN','PTN','CMPf']+NUCLEI:
  for N in NUCLEI:
    for i in range(nbRecord):
      frstr += N+'_'+str(i)+' , '
  frstr+='\n'
  firingRatesFile=open(dataPath+'firingRates.csv','w',1)
  firingRatesFile.write(frstr)

  #-------------------------
  # Simulation
  #-------------------------
  for timeStep in range(nsteps):

    print '====== XP',timeStep+1,'/',nsteps,'======'
    print 'Channel 0:',CSNrate[0,timeStep], PTNrate[0,timeStep]
    print 'Channel 1:',CSNrate[1,timeStep], PTNrate[1,timeStep]

    # reset the network
    nest.ResetNetwork()

    # set up new activities
    frstr = ''
    for Ch in range(2):
      frstr += str(activityLevels[Ch][timeStep]) + ', '
      nest.SetStatus(ActPop['CSN'][Ch],{'rate':CSNrate[Ch,timeStep]})
      nest.SetStatus(ActPop['PTN'][Ch],{'rate':PTNrate[Ch,timeStep]})
    for Ch in range(params['nbCh']):
      nest.SetStatus(ActPop['CMPf'][Ch],{'rate': CMPfFR[1]})

    # offset simulation to bring the system to equilibrium
    nest.Simulate(offsetDuration)
    tstart = str(nest.GetKernelStatus('time'))

    # stores the previous number of spikes
    for N in ['CSN','PTN','CMPf']+NUCLEI:
      for i in range(nbRecord):
        last_n_events[N][i] = nest.GetStatus(inspector[N][i], keys='n_events')[0]
    
    # "real" simulation
    nest.Simulate(simDuration)
    tend = str(nest.GetKernelStatus('time'))
    frstr += tstart + ' , ' + tend + ' , '

    # log the data
    #for N in ['CSN','PTN','CMPf']+NUCLEI:
    for N in NUCLEI:
      for i in range(nbRecord):
        frstr += str((nest.GetStatus(inspector[N][i], keys='n_events')[0] - last_n_events[N][i])/float(simDuration)*1000./nbSim[N])+' , '

    frstr+='\n'
    firingRatesFile.write(frstr)
    firingRatesFile.flush(); os.fsync(firingRatesFile.fileno())

    # log rotation
    for logfile in glob.glob(r'log/*.gdf'):
      shutil.move(logfile, logfile+'_'+str(timeStep)+'.keep')

  firingRatesFile.close()



#-----------------------------------------------------------------------
def main():
  nest.set_verbosity("M_WARNING")

  CSNFRrest = 2.
  PTNFRrest = 15.
  CMPfFRrest = 4.

  # CSN - we expect increase of about 17.7 Hz (Turner 2000)
  # PTN - we expect increase of about 31.3 Hz (Turner 2000)
  # CMPf - we expect increase of up to 30 Hz (Matsumoto 2001)
  CSNFRampl = 17.7
  PTNFRampl = 31.3
  CMPfFRampl = 30.


  spread_factor = 1.
  CSNFRmax = CSNFRrest + CSNFRampl * spread_factor
  PTNFRmax = PTNFRrest + PTNFRampl * spread_factor
  # new version
  proportionCSN = 0.05649718 / spread_factor
  proportionPTN = 0.4792332 / spread_factor

  # adjusted CM/Pf inputs
  proportionCMPf = 0.05 * 0. # no CM/Pf
  #proportionCMPf = 0.05 *  0.5
  #proportionCMPf = 0.05 *  1.
  #proportionCMPf = 0.05 * -1.
  #proportionCMPf = 0.05 * -0.5

  proportionCMPf = 0.05 *  0.

  #proportionCMPf = 0.05 *  3. # very high CM/Pf
  #proportionCMPf = 0.05 *  5. # very high CM/Pf
  #proportionCMPf = 0.05 *  10. # very high CM/Pf
  #proportionCMPf = 0.05 *  15. # very high CM/Pf
  #proportionCMPf = 0.05 *  20. # maximal CM/Pf

  #proportionCMPf = -0.05 *  20. # minimal CM/Pf

  # Proportions are expressed as a fraction of the 12,000 neurons
  # Given that any input population above 4,000 neurons is wonky, there is no point in trying to set it to a higher value than 0.3
  proportionCSN = 0.3 * params['CSNFRmod']
  proportionPTN = 0.3 * params['PTNFRmod']
  proportionCMPf = 0.3 * params['CMPfFRmod']

  sum_constraint = None # full grid search
  #sum_constraint = 1 # only the diagonal
 
  step = 0.1 # non-precise search
  #step = 0.025 # precise (early diagonal study)
  #step = 0.01 # very precise

  instantiate_BG(params, antagInjectionSite='none', antag='')

  CMPfFRmax = CMPfFRampl + CMPfFRrest

  checkSelec(params=params, PActiveCSN=proportionCSN, PActivePTN=proportionPTN, PActiveCMPf=proportionCMPf, CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4., CMPfFRmax], transientCMPf=1., step=step)
  #checkSelec(params=params, PActiveCSN=proportionCSN, PActivePTN=proportionPTN, PActiveCMPf=proportionCMPf, CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4., CMPfFRmax], transientCMPf=1., step=step, sum_constraint=sum_constraint, start=0., end=1.)



#---------------------------
if __name__ == '__main__':
  main()
