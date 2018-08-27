#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## testGPR01.py
##
## This script initializes a single-channel basal ganglia and increases progressively the cortical and thalamic inputs while recording activity in all basal ganglia nuclei, in order to normalize the inputs

from iniBG import *


#-----------------------------------------------------------------------
# params: a parameterization defining `varying` (the name of the nucleus to vary), `PActive_ini` (the initial proportion of neurons recruited in this nucleus), and `PActive_max` (max. prop.)
# CSNFR, PTNFR, CMPfFR: arrays with two items, first the rest firing rate, then the maximal firing rate
#-----------------------------------------------------------------------
def normalizeFiringRate(params, CSNFR, PTNFR, CMPfFR):

  nest.ResetNetwork()
  initNeurons()

  dataPath='log/'
  nest.SetKernelStatus({"overwrite_files":True}) # when we redo the simulation, we erase the previous traces

  varying = params['varying']
  PActive_ini = params['PActive_ini']
  PActive_max = params['PActive_max']

  offsetDuration = 1000. # preliminary period (not recorded)
  if 'simDuration' not in params.keys():
    simDuration = 10000. # overall duration of XP
  else:
    simDuration = params['simDuration']

  if params['nbCh'] != 1:
    raise(KeyError('this experiment works with single-channel nucleus (`nbCh == 1`), aborting.'))

  #-------------------------
  # prepare the input proportions
  #-------------------------  

  activation_levels = np.linspace(PActive_ini, PActive_max, num=simDuration)

  nsteps = len(activation_levels)

  #-------------------------
  # and prepare the lists of neurons that will be affected by these activity changes
  #-------------------------
  ActPop = {'CSN':(),'PTN':(),'CMPf':()}
  def activate_pop(N, PActive, nbCh=2):
    src = Pop[n]
    if 'Fake' in globals():
      if N in Fake:
        src = Fake[N]
    if PActive==1.:
     ActPop[N]=src
    else:
      ActPop[N] = tuple(np.array(src)[range(int(np.ceil(nbSim[N]*PActive)) - 1)]) # activate the first indices: randomizing here would only create noise

  #-------------------------
  # log-related variales
  #-------------------------

  inspector = {}
  for N in ['CSN','PTN','CMPf','MSN','FSI','STN','GPe','GPi']:
    inspector[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": N, "to_file": True, 'start':offsetDuration ,'stop':offsetDuration+simDuration})
    nest.Connect(Pop[N],inspector[N])

  #--------------------------
  # stimulation without inputs, to make sure that the NN is at rest
  #--------------------------
  nest.Simulate(offsetDuration)

  #----------------------------------
  # Loop over the n steps of the test
  #----------------------------------
  for timeStep in range(nsteps):

    if varying == 'CSN':
      activate_pop('CSN', activation_levels[timeStep])
      nest.SetStatus(ActPop['CSN'],{'rate':CSNFR[1]})
    elif varying == 'PTN':
      activate_pop('PTN', activation_levels[timeStep])
      nest.SetStatus(ActPop['PTN'],{'rate':PTNFR[1]})
    elif varying == 'CMPf':
      activate_pop('CMPf', activation_levels[timeStep])
      nest.SetStatus(ActPop['CMPf'],{'rate':CMPfFR[1]})
    elif varying == 'all':
      activate_pop('CSN', activation_levels[timeStep])
      nest.SetStatus(ActPop['CSN'],{'rate':CSNFR[1]})
      activate_pop('PTN', activation_levels[timeStep])
      nest.SetStatus(ActPop['PTN'],{'rate':PTNFR[1]})
      activate_pop('CMPf', activation_levels[timeStep])
      nest.SetStatus(ActPop['CMPf'],{'rate':CMPfFR[1]})
    else:
      raise(KeyError('`varying` must be one of `CSN`, `PTN`, `CMPf`, or `all`, aborting.'))

    #-------------------------
    # Simulation
    #-------------------------
    if timeStep % 100 == 0:
      print('Step '+str(timeStep)+': enabling '+str(np.round(activation_levels[timeStep]*100))+'% neurons in '+varying+'\n')

    nest.Simulate(1)

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

  #proportionCSN = 0.05649718
  #proportionPTN = 0.4792332
  #proportionCMPf = 0.05 # Base proportion of active CM/Pf neurons hypothesized
 
  # for the screening experiment
  if 'varying' not in params.keys():
    raise(KeyError('should specifying `varying`, aborting.'))

  # enforce some defaults
  if 'PActive_ini' not in params.keys():
    params['PActive_ini'] = 0.
  if 'PActive_max' not in params.keys():
    params['PActive_max'] = 1.

  instantiate_BG(params, antagInjectionSite='none', antag='')

  normalizeFiringRate(params=params, CSNFR=[CSNFRrest, CSNFRampl + CSNFRrest], PTNFR=[PTNFRrest, PTNFRampl + PTNFRrest], CMPfFR=[CMPfFRrest, CMPfFRampl + CMPfFRrest])


#---------------------------
if __name__ == '__main__':
  main()
