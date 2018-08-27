#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## testGPR01.py
##
## This script implements the selection test from Gurney, Prescott and Redrave, 2001b

from iniBG import *


#-----------------------------------------------------------------------
# PActiveCNS/PTN : proportion of "active" neurons in the CSN/PTN populations (in [0.,1.])
#
#-----------------------------------------------------------------------
def checkSelec(params={},CSNFR=[2.,10.], PActiveCSN=1., PTNFR=[15.,35], PActivePTN=1., antagInjectionSite='none',antag='', CMPfFR=[4., 4.], PActiveCMPf=1., transientCMPf=1., simDuration=1000., activityLevels = None):

  nest.ResetNetwork()
  initNeurons()

  dataPath='log/'
  nest.SetKernelStatus({"overwrite_files":True}) # when we redo the simulation, we erase the previous traces

  offsetDuration = 1000. # preliminary period (not recorded)
  if 'simDuration' not in params.keys():
    simDuration = 1000. # step duration period
  else:
    simDuration = params['simDuration']

  nbRecord=2 # number of channels whose activity will be recorded
  if params['nbCh'] != 2:
    print 'need exactly 2 channels to perform the selection test'
    exit()

  #-------------------------
  # prepare the firing rates of the inputs for the 3 steps of the experiment
  #-------------------------  
  gCSN = CSNFR[1]-CSNFR[0]
  gPTN = PTNFR[1]-PTNFR[0]
  #gCMPf = CMPfFR[1]-CMPfFR[0]

  if 'selecRamp' not in params.keys():
    # default sim in 3 timesteps
    # channel 1: _--
    # channel 2: __-
    activityLevels = np.array([[0.,1.,1.], [0.,0.,1.]]) 
  elif params['selecRamp'] == 1 or params['selecRamp'] == True:
    # Ramping activity on Channel 2, Channel 1 always at the max, increasing over 10 steps
    # channel 1: ⁻⁻⁻⁻⁻⁻⁻⁻⁻⁻
    # channel 2: ___----⁻⁻⁻
    activityLevels = np.array([
     np.concatenate([np.repeat(1,10), np.array([1])]),
     np.concatenate([np.arange(0,1,0.1), np.array([1])])
    ])
  elif params['selecRamp'] == 0:
    # Ramping activity on Channel 1, Channel 2 alway at baseline, increasing over 10 steps
    # channel 1: ___----⁻⁻⁻
    # channel 2: __________
    activityLevels = np.array([
     np.concatenate([np.arange(0,1,0.1), np.array([1])]),
     np.concatenate([np.repeat(0,10), np.array([1])])
    ])


  nsteps = len(activityLevels[0])
  CSNrate = gCSN * activityLevels + np.ones((nsteps)) * CSNFR[0]
  PTNrate = gPTN * activityLevels + np.ones((nsteps)) * PTNFR[0]
  #CMPfrate = gCMPf * activityLevels + np.ones((nsteps)) * CMPfFR[0]

  #-------------------------
  # and prepare the lists of neurons that will be affected by these activity changes
  #-------------------------
  ActPop = {'CSN':[(),()],'PTN':[(),()],'CMPf':[(),()]}
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
  activate_pop('CMPf', PActiveCMPf)

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

  inspector = {}
  for N in ['CSN','PTN','CMPf']+NUCLEI:
    inspector[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+N, "to_file": True, 'start':0. ,'stop':offsetDuration+simDuration*nsteps})
    for i in range(nbRecord):
      nest.Connect(Pop[N][i],inspector[N])
  #last_n_events = {} # used to compute firing rates
  #for N in ['CSN','PTN','CMPf']+NUCLEI:
  #  inspector[N] = [None for i in range(nbRecord)] # channel-specific recording
  #  last_n_events[N] = [0. for i in range(nbRecord)] # channel-specific firing rate storing
  #  for i in range(nbRecord):
  #    inspector[N][i] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": N+'_'+str(i), "to_file": True, 'start':0. ,'stop':(offsetDuration+simDuration)*nsteps})
  #    nest.Connect(Pop[N][i],inspector[N][i])

  #--------------------------
  # stimulation without inputs, to make sure that the NN is at rest
  #--------------------------
  nest.Simulate(offsetDuration)

  #----------------------------------
  # Loop over the n steps of the test
  #----------------------------------
  for timeStep in range(nsteps):

    frstr = str(timeStep) + ', '

    #-------------------------
    # Simulation
    #-------------------------
    print '====== Step',timeStep,'======'
    print 'Channel 0:',CSNrate[0,timeStep], PTNrate[0,timeStep], CMPfFR[1]
    print 'Channel 1:',CSNrate[1,timeStep], PTNrate[1,timeStep], CMPfFR[1]

    nest.SetStatus(ActPop['CSN'][0],{'rate':CSNrate[0,timeStep]})
    nest.SetStatus(ActPop['CSN'][1],{'rate':CSNrate[1,timeStep]})
    nest.SetStatus(ActPop['PTN'][0],{'rate':PTNrate[0,timeStep]})
    nest.SetStatus(ActPop['PTN'][1],{'rate':PTNrate[1,timeStep]})
    for Ch in range(params['nbCh']):
      nest.SetStatus(ActPop['CMPf'][Ch],{'rate': CMPfFR[1]})
    #nest.SetStatus(ActPop['CMPf'][0],{'rate':CMPfrate[0,timeStep]})
    #nest.SetStatus(ActPop['CMPf'][1],{'rate':CMPfrate[1,timeStep]})

    nest.Simulate(simDuration)
    #if PActiveCMPf == 0. or CMPfFR[0] == CMPfFR[1] or transientCMPf == 0.:
    #  # no CMPf
    #  nest.Simulate(simDuration)
    #else:
    #  # CMPf activity during selection
    #  print('CMPf activity increased to ' + str(CMPfFR[1]) + ' for ' + str((simDuration+offsetDuration)*transientCMPf) + ' ms\n')
    #  for Ch in range(params['nbCh']):
    #    nest.SetStatus(ActPop['CMPf'][Ch],{'rate': CMPfFR[1]})
    #  nest.Simulate(simDuration*transientCMPf)
    #  for Ch in range(params['nbCh']):
    #    nest.SetStatus(ActPop['CMPf'][Ch],{'rate': CMPfFR[0]})
    #  if transientCMPf < 1.:
    #   nest.Simulate(simDuration*(1.-transientCMPf))
        

#-----------------------------------------------------------------------
def main():
  nest.set_verbosity("M_WARNING")

  # old way of simulating inputs:
  #CSNFRmax = 2. + 0.5 * params['CSNFRmod']
  #PTNFRmax = 15. + 15. * params['PTNFRmod']
  #CMPfFRmax = 4. + 1.5 * CMPfFRmod

  ## new way: starting from expecting external inputs, then taking ratios
  ## CSN - we expect increase of about 17.7 Hz (Turner 2000)
  ## PTN - we expect increase of about 31.3 Hz (Turner 2000)
  ## CMPf - we expect increase of up to 30 Hz (Matsumoto 2001)
  #CSNFRmax = 2. + 17.7
  #PTNFRmax = 15. + 31.3
  #CMPfFRmax = 4. + 30.
  #proportionCSN = 0.02824859
  #proportionPTN = 0.4792332
  #proportionCMPf = 0.05

  ############################
  # imported from testSelec2D
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

  # hand-tuned first range
  proportionCSN = 0.05649718 / spread_factor
  proportionPTN = 0.4792332 / spread_factor
  proportionCMPf = 0.05 # Base proportion of active CM/Pf neurons hypothesized

  # broader range of 30% (up to 4000 neurons) from the single-input systematic variation (paper figure 3)
  proportionCSN = 0.3
  proportionPTN = 0.3
  proportionCMPf = 0.3
 
  # for the screening experiment
  if 'CSNFRmod' in params.keys() and 'PTNFRmod' in params.keys():
    if 'CMPfFRmod' in params.keys():
      # use the supplied CMPf input ratio
      CMPfFRmod = params['CMPfFRmod']
    else:
      # if CMPf not supplied: assume ternary simulations (all inputs sum to 100%)
      CMPfFRmod = 1. - params['CSNFRmod'] - params['PTNFRmod']
      if params['CSNFRmod'] + params['PTNFRmod'] > 1.:
        print('would not do this simulation in ternary:\n'+
          'CSNFRmod = '+str(params['CSNFRmod'])+'\n',
          'PTNFRmod = '+str(params['PTNFRmod'])+'\n',
          'CMPfFRmod = '+str(CMPfFRmod)+'\n',
        )
      return
    print('\n\nnow doing this simulation:\n'+
      'CSNFRmod = '+str(params['CSNFRmod'])+'\n'+
      'PTNFRmod = '+str(params['PTNFRmod'])+'\n'+
      'CMPfFRmod = '+str(CMPfFRmod)+'\n\n'
    )
    paramsFile = open('ternary','w')
    paramsFile.writelines([
      'CSNFRmod, '+str(params['CSNFRmod'])+'\n',
      'PTNFRmod, '+str(params['PTNFRmod'])+'\n',
      'CMPfFRmod, '+str(CMPfFRmod)+'\n',
    ])
    paramsFile.close()
    instantiate_BG(params, antagInjectionSite='none', antag='')
    #checkSelec(params=params, PActiveCSN=proportionCSN*params['CSNFRmod'], PActivePTN=proportionPTN*params['PTNFRmod'], PActiveCMPf=proportionCMPf*CMPfFRmod, CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4.,CMPfFRmax], transientCMPf=1.)

    # From testSelec2D:
    CMPfFRmax = CMPfFRampl + CMPfFRrest
    checkSelec(params=params, PActiveCSN=proportionCSN*params['CSNFRmod'], PActivePTN=proportionPTN*params['PTNFRmod'], PActiveCMPf=proportionCMPf*params['CMPfFRmod'], CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4., CMPfFRmax], transientCMPf=1.)
  else:
    instantiate_BG(params, antagInjectionSite='none', antag='')

    print("Update code from testSelec2D before running...")
    #checkSelec(showRasters=True, params=params, PActiveCSN=proportionCSN, PActivePTN=proportionPTN, PActiveCMPf=proportionCMPf, CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4.,CMPfFRmax], transientCMPf=1.)

    # OK pure CSN input - 2018_03_13_15_27_08_xp000000_selec
    #checkSelec(showRasters=True,params=params,CSNFR=[2.,2.5], PTNFR=[15.,15], CMPfFR=[4., 4.])

    # OK pure PTN input - 2018_03_14_14_18_33_xp000000_selec
    #checkSelec(showRasters=True,params=params,CSNFR=[2.,2.], PTNFR=[15.,30.], CMPfFR=[4., 4.])

    # OK pure CMPf input - 2018_03_14_14_42_01_xp000000_selec
    #checkSelec(showRasters=True,params=params,CSNFR=[2.,2.], PTNFR=[15.,15.], CMPfFR=[4., 5.5])


#---------------------------
if __name__ == '__main__':
  main()
