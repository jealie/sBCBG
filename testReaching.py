#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## testGPR01.py
##
## This script implements the arm reaching task from Georgopoulos et al., 1982

from iniBG import *


#-----------------------------------------------------------------------
# PActiveCNS/PTN : proportion of "active" neurons in the CSN/PTN populations (in [0.,1.])
#
#-----------------------------------------------------------------------
def checkSelec(showRasters=False,params={},CSNFR=[2., 10.], PActiveCSN=1., PTNFR=[15.,35], PActivePTN=1., antagInjectionSite='none',antag='', CMPfFR=[4., 4.], PActiveCMPf=1., transientCMPf=1.):

  nest.ResetNetwork()
  initNeurons()

  dataPath='log/'
  nest.SetKernelStatus({"overwrite_files":True}) # when we redo the simulation, we erase the previous traces

  # 'long' simulations
  #offsetDuration = 3000. # preliminary period (not recorded)
  #simDuration = 3000. # duration of one simulation step

  # 'short' ones
  offsetDuration = 1000. # preliminary period (not recorded)
  simDuration = 2000. # duration of one simulation step

  nbRecord=8 # number of channels whose activity will be recorded
  if params['nbCh'] != 8:
    print 'need exactly 8 channels to perform the arm reaching task'
    exit()

  #-------------------------
  # prepare the firing rates of the inputs
  #-------------------------  

  ## Maximal firing rate
  gCSN = CSNFR[1]-CSNFR[0]
  gPTN = PTNFR[1]-PTNFR[0]
  gCMPf = CMPfFR[1]-CMPfFR[0]
  maxChangeRel = np.ones((params['nbCh']))
  for i in range(params['nbCh']):
    #maxChangeRel[i] = 2. * np.pi / (params['nbCh']) * i
    maxChangeRel[i] = 2. * np.pi / (params['nbCh']) * i
  maxChangeRel = (np.cos(maxChangeRel)+1)/2.
  #maxChangeRel = maxChangeRel * (1.+baselevel) - baselevel
  CSNrate =   gCSN * maxChangeRel
  PTNrate =   gPTN * maxChangeRel
  CMPfrate = gCMPf * maxChangeRel # focused
  #CMPfrate = gCMPf * np.ones((params['nbCh'])) # diffused

  ## Minimal firing rate
  baselevel = 0. # LG14: only increases, no decrease of activity
  baselevel = 0.2 # the activity of opposite direction decrease by ~20% of the baseline level (cf. Kalaska et al., but the decrease of activity is under-estimated because averaged over a long period that encompasses a lot of non-task time)
  baselevel = 2.0 # Best visual fit to visual plots of Kalaska et al and Georgopoulos: the activity of opposite direction actually decrease and reaches silence in the 3 farthest channels
  CSNdip  = (1. - baselevel) * CSNFR[0] # lowest FR of inputs
  PTNdip  = (1. - baselevel) * PTNFR[0]
  CMPfdip = (1. - baselevel) * CMPfFR[0]
  CSNdip  =  baselevel * CSNFR[0] # lowest FR of inputs
  PTNdip  =  baselevel * PTNFR[0]
  CMPfdip =  baselevel * CMPfFR[0]
  CSNrate =  CSNrate * (1. + CSNdip/gCSN)- CSNdip
  PTNrate =  PTNrate * (1. + PTNdip/gPTN)- PTNdip
  CMPfrate = CMPfrate * (1. + CMPfdip/gCMPf)- CMPfdip

  # steps is in the form:
  # step_nb: [duration, percent_max_change]
  steps = {1: [790., 0.]}                     # 750 ms
  for i in range(5):
    steps[i+2] = [10., (i+1)/5.]              # 800 ms
  steps[7] = [270., 1.]                       # 950 ms
  for i in range(5):
    steps[i+8] = [10., (5-i)/5.]              # 1000 ms
  steps[13] = [840., 0.]                     # 2000 ms

  #-------------------------
  # and prepare the lists of neurons that will be affected by these activity changes
  #-------------------------
  ActPop = {'CSN':  [(),(),(),(),(),(),(),()],
            'PTN':  [(),(),(),(),(),(),(),()],
           'CMPf':  [(),(),(),(),(),(),(),()]}
  def activate_pop(N, PActive, nbCh=8):
    src = Pop[N]
    if 'Fake' in globals():
      if N in Fake:
        src = Fake[N]
    if PActive==1.:
      ActPop[N]=src
    else:
      for i in range(nbCh):
        #ActPop[N][i] = tuple(rnd.choice(a=np.array(src[i]),size=int(np.ceil(nbSim[N]*PActive)),replace=False)) # random sub-population
        ActPop[N][i] = tuple(np.array(src[i])[range(int(np.ceil(nbSim[N]*PActive)) - 1)]) # first indices
  activate_pop('CSN', PActiveCSN)
  activate_pop('PTN', PActivePTN)
  activate_pop('CMPf', PActiveCMPf)

  #-------------------------
  # log-related variables
  #-------------------------

  antagStr = ''
  if antagInjectionSite != 'none':
    antagStr = antagInjectionSite+'_'+antag+'_'

  inspector = {}
  for N in NUCLEI+['CSN','PTN','CMPf']:
    inspector[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+N, "to_file": True, 'start':offsetDuration ,'stop':offsetDuration+simDuration})
    for i in range(nbRecord):
      nest.Connect(Pop[N][i],inspector[N])

  # stimulation without inputs, to make sure that the NN is at rest
  nest.Simulate(offsetDuration)

  #----------------------------------
  # Loop over the 3 steps of the test
  #----------------------------------
  for timeStep in range(len(steps)):

    frstr = str(timeStep) + ', '

    stepDuration = steps[timeStep+1][0]
    stepModulation = steps[timeStep+1][1]

    #-------------------------
    # Simulation
    #-------------------------
    print '====== Step',timeStep,'======'
    print('step duration: '+str(stepDuration))
    print('step modulation: '+str(stepModulation))
    stepCSN = CSNFR[0] + CSNrate * stepModulation
    stepPTN = PTNFR[0] + PTNrate * stepModulation
    stepCMPf = CMPfFR[0] + CMPfrate * stepModulation
    for i in range(params['nbCh']):
      stepCSN_i = max(0., stepCSN[i])
      stepPTN_i = max(0., stepPTN[i])
      stepCMPf_i = max(0., stepCMPf[i])
      print 'Channel ',str(i),':',stepCSN_i, stepPTN_i, stepCMPf_i
      nest.SetStatus(ActPop['CSN'][i], {'rate': stepCSN_i})
      nest.SetStatus(ActPop['PTN'][i], {'rate': stepPTN_i})
      nest.SetStatus(ActPop['CMPf'][i], {'rate': stepCMPf_i})

    nest.Simulate(stepDuration)
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

  # new way: starting from expecting external inputs, then taking ratios
  # CSN - we expect increase of about 17.7 Hz (Turner 2000)
  # PTN - we expect increase of about 31.3 Hz (Turner 2000)
  # CMPf - we expect increase of up to 30 Hz (Matsumoto 2001)
  CSNFRmax = 2. + 17.7 * 2.
  PTNFRmax = 15. + 31.3 * 2.
  CMPfFRmax = 4. + 30. * 2.
  proportionCSN = 0.02824859 / 2.
  proportionPTN = 0.4792332 / 2.
  proportionCMPf = 0.05 / 2.
 
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
    checkSelec(showRasters=True, params=params, PActiveCSN=proportionCSN*params['CSNFRmod'], PActivePTN=proportionPTN*params['PTNFRmod'], PActiveCMPf=proportionCMPf*CMPfFRmod, CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4.,CMPfFRmax], transientCMPf=1.)
  else:
    instantiate_BG(params, antagInjectionSite='none', antag='')

    checkSelec(showRasters=True, params=params, PActiveCSN=proportionCSN, PActivePTN=proportionPTN, PActiveCMPf=proportionCMPf, CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4.,CMPfFRmax], transientCMPf=1.) # 'DiffusedActive' & 'Focused'
    #checkSelec(showRasters=True, params=params, PActiveCSN=proportionCSN, PActivePTN=proportionPTN, PActiveCMPf=proportionCMPf, CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4.,4.], transientCMPf=1.) # 'neutralCMPf'
    #checkSelec(showRasters=True, params=params, PActiveCSN=proportionCSN, PActivePTN=proportionPTN, PActiveCMPf=proportionCMPf, CSNFR=[2.,CSNFRmax], PTNFR=[15.,PTNFRmax], CMPfFR=[4.,0.05], transientCMPf=1.) # 'antiCMPf'

    # OK pure CSN input - 2018_03_13_15_27_08_xp000000_selec
    #checkSelec(showRasters=True,params=params,CSNFR=[2.,2.5], PTNFR=[15.,15], CMPfFR=[4., 4.])

    # OK pure PTN input - 2018_03_14_14_18_33_xp000000_selec
    #checkSelec(showRasters=True,params=params,CSNFR=[2.,2.], PTNFR=[15.,30.], CMPfFR=[4., 4.])

    # OK pure CMPf input - 2018_03_14_14_42_01_xp000000_selec
    #checkSelec(showRasters=True,params=params,CSNFR=[2.,2.], PTNFR=[15.,15.], CMPfFR=[4., 5.5])


#---------------------------
if __name__ == '__main__':
  main()
