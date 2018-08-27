#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## testGPR01Deact.py
##
## This script implements the selection test from Gurney, Prescott and Redrave, 2001b, while deactivating

from iniBG import *


def restRun(params={}):
  nest.ResetNetwork()
  initNeurons()

  dataPath='log/'
  nest.SetKernelStatus({"overwrite_files":True}) # when we redo the simulation, we erase the previous traces

  if 'simDuration' not in params.keys():
    simDuration = 5000. # step duration period
  else:
    simDuration = params['simDuration'] * 5.
    #simDuration = 1000. # step duration period
  offsetDuration = simDuration / 5. # preliminary period (not recorded)
  #offsetDuration = simDuration # preliminary period (not recorded)
  
  inspector = {}
  for N in ['CSN','PTN','CMPf']+NUCLEI:
    inspector[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": 'rest'+N, "to_file": True, 'start':offsetDuration+nest.GetKernelStatus('time'),'stop':offsetDuration+simDuration+nest.GetKernelStatus('time')})
    for i in range(params['nbCh']):
      nest.Connect(Pop[N][i],inspector[N])

  # stimulation without inputs, to make sure that the NN is at rest
  nest.Simulate(offsetDuration+simDuration)

  restRate={}
  for N in ['CSN','PTN','CMPf']+NUCLEI:
    restRate[N] = nest.GetStatus(inspector[N], 'n_events')[0] / float(nbSim[N]*simDuration*params['nbCh']) * 1000
  #import ipdb; ipdb.set_trace()
  return restRate

#-----------------------------------------------------------------------
# PActiveCNS/PTN/CMPf : proportion of "active" neurons in the CSN/PTN/CMPf populations (in [0.,1.])
#-----------------------------------------------------------------------
def checkSelec(params={}, CSNFR=[2.,10.], PActiveCSN=1., PTNFR=[15.,35], PActivePTN=1., CMPfFR=[4., 4.], PActiveCMPf=1., transientCMPf=1., simDuration=1000., deactStr='', activityLevels = None):

  nest.ResetNetwork()
  initNeurons()

  dataPath='log/'
  nest.SetKernelStatus({"overwrite_files":True}) # when we redo the simulation, we erase the previous traces

  if 'simDuration' not in params.keys():
    simDuration = 1000. # step duration period
  else:
    simDuration = params['simDuration']
  offsetDuration = simDuration # preliminary period (not recorded)

  if params['nbCh'] < 2:
    print 'need at least 2 channels to perform the selection test'
    exit()
  nbRecord = min(3, params['nbCh']) # number of channels whose activity will be recorded

  #-------------------------
  # prepare the firing rates of the inputs for the 3 steps of the experiment
  #-------------------------  
  gCSN = CSNFR[1]-CSNFR[0]
  gPTN = PTNFR[1]-PTNFR[0]
  gCMPf = CMPfFR[1]-CMPfFR[0]

  if 'selecRamp' not in params.keys():
    # simulation in the 5-step GPR test
    # channel 1: _--⁻-
    # channel 2: __⁻⁻⁻
    activityLevels = np.array([[0.,0.4,0.4,1.,0.4], [0.,0.,1.,1.,1.]]) 
    ## default sim in 3 timesteps
    ## channel 1: _--
    ## channel 2: __-
    #activityLevels = np.array([[0.,1.,1.], [0.,0.,1.]]) 
  else:
    activityLevels = np.array([
     np.concatenate([np.repeat(1,10), np.array([1])]),
     np.concatenate([np.arange(0,1,0.1), np.array([1])])
    ]) 

  nsteps = len(activityLevels[0])
  CSNrate = gCSN * activityLevels + np.ones((nsteps)) * CSNFR[0]
  PTNrate = gPTN * activityLevels + np.ones((nsteps)) * PTNFR[0]
  CMPfrate = gCMPf * activityLevels + np.ones((nsteps)) * CMPfFR[0]

  #-------------------------
  # and prepare the lists of neurons that will be affected by these activity changes
  #-------------------------
  ActPop = {'CSN':[(), ()], 'PTN':[(), ()], 'CMPf':[() for i in range(params['nbCh'])]}
  def activate_pop(N, PActive, nbCh=2):
    src = Pop[n]
    if 'Fake' in globals():
      if N in Fake:
        src = Fake[N]
    if PActive == 0.:
      pass # nothing to connect
    elif PActive == 1.:
      for i in range(nbCh):
        ActPop[N][i] = src[i] # connect all
    else:
      for i in range(nbCh):
        ActPop[N][i] = tuple(rnd.choice(a=np.array(src[i]),size=int(nbSim[N]*PActive),replace=False))
  activate_pop('CSN', PActiveCSN)
  activate_pop('PTN', PActivePTN)
  activate_pop('CMPf', PActiveCMPf, nbCh=params['nbCh']) # CMPf activated on all channels

  #-------------------------
  # log-related variales
  #-------------------------
  score = 0
  expeRate={}
  for N in NUCLEI:
    expeRate[N]=-1. * np.ones((nbRecord,3))

  inspector = {}
  for N in ['CSN','PTN','CMPf']+NUCLEI:
    inspector[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": deactStr+N, "to_file": True, 'start': nest.GetKernelStatus('time') ,'stop': offsetDuration+simDuration*nsteps+nest.GetKernelStatus('time')})
    for i in range(nbRecord):
      nest.Connect(Pop[N][i],inspector[N])

  # stimulation without inputs, to make sure that the NN is at rest
  nest.Simulate(offsetDuration)

  #----------------------------------
  # Loop over the 3 steps of the test
  #----------------------------------
  for timeStep in range(nsteps):

    frstr = str(timeStep) + ', '

    #-------------------------
    # Simulation
    #-------------------------
    print '====== Step',timeStep,'======'
    print 'Channel 0:',CSNrate[0,timeStep], PTNrate[0,timeStep], CMPfrate[0,timeStep]
    print 'Channel 1:',CSNrate[1,timeStep], PTNrate[1,timeStep], CMPfrate[1,timeStep]

    nest.SetStatus(ActPop['CSN'][0],{'rate':CSNrate[0,timeStep]})
    nest.SetStatus(ActPop['CSN'][1],{'rate':CSNrate[1,timeStep]})
    nest.SetStatus(ActPop['PTN'][0],{'rate':PTNrate[0,timeStep]})
    nest.SetStatus(ActPop['PTN'][1],{'rate':PTNrate[1,timeStep]})
    for i in range(params['nbCh']):
      nest.SetStatus(ActPop['CMPf'][i],{'rate':CMPfrate[0,timeStep]})

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
  CMPfFRmax = CMPfFRrest + CMPfFRampl * spread_factor
  # new version
  proportionCSN = 0.05649718 / spread_factor
  proportionPTN = 0.4792332 / spread_factor
  proportionCMPf = 0.05 * 0. # no CM/Pf
  #proportionCMPf = 0.05 * 0.5
  #proportionCMPf = 0.05 * 1.
  #proportionCMPf = 0.05 * -1.
  #proportionCMPf = 0.05 * -0.5
  ## original spread version
  #proportionCSN = 0.02824859 / spread_factor
  #proportionPTN = 0.4792332 / spread_factor
  #proportionCMPf = 0.05 / spread_factor

  # fpr the screening experiment
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
    checkSelec(params=params, PActiveCSN=proportionCSN*params['CSNFRmod'], PActivePTN=proportionPTN*params['PTNFRmod'], PActiveCMPf=proportionCMPf*CMPfFRmod, CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4.,CMPfFRmax], transientCMPf=1.)
  else:
    instantiate_BG(params, antagInjectionSite='none', antag='')

    # CMPf is not time-dependent
    CMPfFRmax = proportionCMPf * CMPfFRampl + CMPfFRrest
    CMPfFRrest = CMPfFRmax
    proportionCMPf = 1.

    print("Establishing baseline activities...")
    baseFR = restRun(params)
    #baseFR={'GPe': 62.266666666666666, 'GPi': 62.30952380952381, 'MSN': 0.3785930408472012, 'FSI': 11.534591194968554, 'STN': 20.416666666666668} # saved firing rates for params 9 (with scale=4)
    baseFR['CSN'] = 2. # enforcing the most precise values in input
    baseFR['PTN'] = 15.
    baseFR['CMPf'] = 4.
    print("Done. Baseline: "+str(baseFR))

    print("Normal selection run...")
    checkSelec(params=params, PActiveCSN=proportionCSN, PActivePTN=proportionPTN, PActiveCMPf=proportionCMPf, CSNFR=[CSNFRrest, CSNFRmax], PTNFR=[PTNFRrest, PTNFRmax], CMPfFR=[CMPfFRrest, CMPfFRmax], transientCMPf=1.)

    #########################################
    # Deactivate each nucleus one at a time #
    #########################################
    # nuc_from is the nucleus to bypass
    # relevant only for nuclei that send efference to the BG (ie not GPi)
    for nuc_from in ['CSN','PTN','CMPf','MSN','FSI','STN','GPe']:
      # start with a clean slate
      instantiate_BG(params, antagInjectionSite='none', antag='')
      # set the fake nucleus parameters as the original one
      Poisson_copy(nuc_from, 'Fake_'+nuc_from, baseFR[nuc_from])
      # relevant only for BG nuclei (ie not CSN, PTN or CM/Pf)
      for nuc_to in NUCLEI:
        if nuc_from+'->'+nuc_to not in P.keys():
          print('skipping non-existent connection '+nuc_from+'->'+nuc_to)
          continue
        # bypass all connections from this nucleus
        bypass_connection(nuc_from, 'Fake_'+nuc_from, nuc_to)
      # run the simulation
      checkSelec(params=params, PActiveCSN=proportionCSN, PActivePTN=proportionPTN, PActiveCMPf=proportionCMPf, CSNFR=[CSNFRrest, CSNFRmax], PTNFR=[PTNFRrest, PTNFRmax], CMPfFR=[CMPfFRrest, CMPfFRmax], transientCMPf=1., deactStr = nuc_from+'_all_')

    ############################################
    # Deactivate each connection one at a time #
    ############################################
    # nuc_from -> nuc_to is the connection to bypass
    # relevant only for nuclei that send efference to the BG (ie not GPi)
    for nuc_from in ['CSN','PTN','CMPf','MSN','FSI','STN','GPe']:
      # relevant only for BG nuclei (ie not CSN, PTN or CM/Pf)
      for nuc_to in NUCLEI:
        if nuc_from+'->'+nuc_to not in P.keys():
          print('skipping non-existent connection '+nuc_from+'->'+nuc_to)
          continue
        # start with a clean slate
        instantiate_BG(params, antagInjectionSite='none', antag='')
        # set the fake nucleus parameters as the original one
        Poisson_copy(nuc_from, 'Fake_'+nuc_from, baseFR[nuc_from])
        # bypass the chosen connection
        bypass_connection(nuc_from, 'Fake_'+nuc_from, nuc_to)
        # run the simulation
        checkSelec(params=params, PActiveCSN=proportionCSN, PActivePTN=proportionPTN, PActiveCMPf=proportionCMPf, CSNFR=[CSNFRrest, CSNFRmax], PTNFR=[PTNFRrest, PTNFRmax], CMPfFR=[CMPfFRrest, CMPfFRmax], transientCMPf=1., deactStr = nuc_from+'_'+nuc_to+'_')

#---------------------------
if __name__ == '__main__':
  main()
