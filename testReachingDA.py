#!/usr/bin/env python
# -*- coding: utf-8 -*-

##
## testPlausibility.py
##
## This script tests the DA implementation on the stripped down version of the BG

from iniBG import *

import pandas as pd

restFR = {} # this will be populated with firing rates of all nuclei, at rest
oscilPow = {} # Oscillations power and frequency at rest
oscilFreq = {}

#------------------------------------------
# Checks whether the BG model respects the electrophysiological constaints (firing rate at rest).
# If testing for a given antagonist injection experiment, specifiy the injection site in antagInjectionSite, and the type of antagonists used in antag.
# Returns [score obtained, maximal score]
# params possible keys:
# - nb{MSN,FSI,STN,GPi,GPe,CSN,PTN,CMPf} : number of simulated neurons for each population
# - Ie{GPe,GPi} : constant input current to GPe and GPi
# - G{MSN,FSI,STN,GPi,GPe} : gain to be applied on LG14 input synaptic weights for each population
#------------------------------------------
def checkAvgFRTst(params, CSNFR, PActiveCSN, PTNFR, PActivePTN, CMPfFR, PActiveCMPf, action=[], antagInjectionSite='none', antag='', showRasters=False):
  #nest.ResetNetwork()
  #initNeurons()

  showPotential = False # Switch to True to graph neurons' membrane potentials - does not handle well restarted simulations

  dataPath='log/'
  nest.SetKernelStatus({"overwrite_files":True}) # when we redo the simulation, we erase the previous traces

  nstrand.set_seed(params['nestSeed'], params['pythonSeed']) # sets the seed for the simulation

  simulationOffset = nest.GetKernelStatus('time')
  print('Simulation Offset: '+str(simulationOffset))
  offsetDuration = 500.
  simDuration = 4000. # ms

  # single or multi-channel?
  if params['nbCh'] == 1:
    connect_detector = lambda N: nest.Connect(Pop[N], spkDetect[N])
    #disconnect_detector = lambda N, _: [nest.DisconnectOneToOne(Pop[N][i], spkDetect[N][0], syn_spec={}) for i in range(len(Pop[N]))]
    #disconnect_detector = lambda N, detector: ipdb.set_trace()
    connect_multimeter = lambda N: nest.Connect(multimeters[N], [Pop[N][0]])
  else:
    connect_detector= lambda N: [nest.Connect(Pop[N][i], spkDetect[N]) for i in range(len(Pop[N]))]
    #disconnect_detector= lambda N: [nest.Disconnect(Pop[N][i], spkDetect[N]) for i in range(len(Pop[N]))]
    connect_multimeter = lambda N: nest.Connect(multimeters[N], [Pop[N][0][0]])

  
  gCSN = CSNFR[1]-CSNFR[0]
  gPTN = PTNFR[1]-PTNFR[0]
  gCMPf = CMPfFR[1]-CMPfFR[0]
  maxChangeRel = np.zeros((params['nbCh']))+0.1
  for a in action:
    maxChangeRel[a] = 1.
  CSNrate =   gCSN * maxChangeRel
  PTNrate =   gPTN * maxChangeRel
  CMPfrate = gCMPf * maxChangeRel

  baselevel = 2.0 # Best visual fit to visual plots of Kalaska et al and Georgopoulos: the activity of opposite direction actually decrease and reaches silence in the 3 farthest channels
  CSNdip  = (1. - baselevel) * CSNFR[0] # lowest FR of inputs
  PTNdip  = (1. - baselevel) * PTNFR[0]
  CMPfdip = (1. - baselevel) * CMPfFR[0]
  CSNdip  =  baselevel * CSNFR[0] # lowest FR of inputs
  PTNdip  =  baselevel * PTNFR[0]
  CMPfdip =  baselevel * CMPfFR[0]
  if gCSN != 0.:
    CSNrate =  CSNrate * (1. + CSNdip/gCSN)- CSNdip
  else:
    CSNrate = np.zeros((params['nbCh']))
  if gPTN != 0.:
    PTNrate =  PTNrate * (1. + PTNdip/gPTN)- PTNdip
  else:
    PTNrate = np.zeros((params['nbCh']))
  if gCMPf != 0.:
    CMPfrate = CMPfrate * (1. + CMPfdip/gCMPf)- CMPfdip
  else:
    CMPfrate = np.zeros((params['nbCh']))

  #-------------------------
  # prepare the lists of neurons that will be affected by the activity changes
  #-------------------------
  #ActPop = {'CSN':  [(),(),(),(),(),(),(),()],
  #          'PTN':  [(),(),(),(),(),(),(),()],
  #         'CMPf':  [(),(),(),(),(),(),(),()]}
  ActPop = {'CSN':  [(),(),(),()],
            'PTN':  [(),(),(),()],
           'CMPf':  [(),(),(),()]}
  def activate_pop(N, PActive, nbCh):
    src = Pop[N]
    if 'Fake' in globals():
      if N in Fake:
        src = Fake[N]
    for i in range(nbCh):
      if PActive==1.:
        ActPop[N][i] = tuple(src[i])
      else:
        #ActPop[N][i] = tuple(rnd.choice(a=np.array(src[i]),size=int(np.ceil(nbSim[N]*PActive)),replace=False)) # random sub-population
        ActPop[N][i] = tuple(np.array(src[i])[range(int(np.ceil(nbSim[N]*PActive)) - 1)]) # first indices
  activate_pop('CSN', PActiveCSN, params['nbCh'])
  activate_pop('PTN', PActivePTN, params['nbCh'])
  activate_pop('CMPf', PActiveCMPf, params['nbCh'])

  #-------------------------
  # measures
  #-------------------------
  spkDetect={} # spike detectors used to record the experiment
  multimeters={} # multimeters used to record one neuron in each population
  expeRate={}

  antagStr = ''
  if antagInjectionSite != 'none':
    antagStr = antagInjectionSite+'_'+antag+'_'

  for N in NUCLEI+['CSN', 'PTN', 'CMPf']:
    # 1000ms offset period for network stabilization
    spkDetect[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": antagStr+N, "to_file": storeGDF, 'start':offsetDuration+simulationOffset,'stop':offsetDuration+simDuration+simulationOffset})
    connect_detector(N)
    if showPotential:
      # multimeter records only the last 200ms in one neuron in each population
      multimeters[N] = nest.Create('multimeter', params = {"withgid": True, 'withtime': True, 'interval': 0.1, 'record_from': ['V_m'], "label": antagStr+N, "to_file": False, 'start':offsetDuration+simulationOffset+simDuration-200.,'stop':offsetDuration+simDuration+simulationOffset})
      connect_multimeter(N)

  #-------------------------
  # Simulation
  #-------------------------

  # rest level
  nest.Simulate(offsetDuration)

  stepModulation = 1. # no temporal modulation: the whole block is uniform
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

  nest.Simulate(simDuration)

  score = 0

  #-------------------------
  # Displays
  #-------------------------
  if showRasters and interactive:
    displayStr = ' ('+antagStr[:-1]+')' if (antagInjectionSite != 'none') else ''
    for N in NUCLEI:
      # histograms crash in the multi-channels case
      nest.raster_plot.from_device(spkDetect[N], hist=(params['nbCh'] == 1), title=N+displayStr)

    if showPotential:
      pl.figure()
      nsub = 231
      for N in NUCLEI:
        pl.subplot(nsub)
        nest.voltage_trace.from_device(multimeters[N],title=N+displayStr+' #0')
        #disconnect_detector(N, spkDetect[N])
        pl.axhline(y=BGparams[N]['V_th'], color='r', linestyle='-')
        nsub += 1
    pl.show()

  #for N in NUCLEI:
  #  disconnect_detector(N, spkDetect[N])

  return [nest.GetStatus(spkDetect['MSN'], keys='events'), nest.GetStatus(spkDetect['CSN'], keys='events')]

def rotate_logs():
  import glob
  import shutil
  for logfile in glob.glob(r'log/*.gdf'):
    shutil.copy(logfile, logfile+'.keep')

def reinforcement_learning(co_spikes, base_weights, DA_level, min_ratio = 0.5, max_ratio = 1.5, viscosity = 0.):
  from tqdm import tqdm
  unique_MSN = np.unique(co_spikes[0][0]['senders'])
  MSN_spikes = pd.DataFrame({'MSNtimes': co_spikes[0][0]['times'], 'MSNneurons': co_spikes[0][0]['senders']})
  MSN_spikes.set_index('MSNtimes', inplace=True, drop=False)
  CSN_spikes = pd.DataFrame({'CSNtimes': co_spikes[1][0]['times'], 'CSNneurons': co_spikes[1][0]['senders']})
  CSN_spikes.set_index('CSNtimes', inplace=True, drop=False)
  for aMSN in tqdm(unique_MSN):
    if params['nbCh'] > 1:
      CSN_IDs = np.array(Pop['CSN']).flatten().tolist()
    else:
      CSN_IDs = Pop['CSN']
    targetingCSN = pd.Series(nest.GetConnections(source=CSN_IDs, target=[aMSN])).apply(lambda x: x[0])
    CSN_spiking_times = CSN_spikes.loc[CSN_spikes['CSNneurons'].isin(targetingCSN)]
    MSN_spiking_times = MSN_spikes.loc[MSN_spikes['MSNneurons'] == aMSN]
    nearest_collision = MSN_spiking_times.reindex(CSN_spiking_times.index, method='nearest')
    nearest_collision['lag'] = nearest_collision['MSNtimes'] - nearest_collision.index # time between CSN and MSN firing (negative lag <=> CSN fired after MSN)
    window = 50.
    nearest_collision = nearest_collision.loc[nearest_collision['lag'].abs() < window] # keep only close collisions
    nearest_collision = nearest_collision.merge(CSN_spiking_times[['CSNneurons']], how='left', left_index=True, right_index=True) # merge back the CSN neuron ID
    # classical hebbian learning (post after pre: strengthening, post before pre: weakening)
    # or something else?
    if DA_level == 1:
      # High DA (mostly classical hebbian)
      if aMSN % 2 == 0:
        # D1 - High DA
        #print("D1 - High DA")
        nearest_collision.loc[nearest_collision['lag']>0, 'impact'] = 1. - nearest_collision.loc[nearest_collision['lag']>0, 'lag']/window
        nearest_collision.loc[nearest_collision['lag']<0, 'impact'] = (- 1. - nearest_collision.loc[nearest_collision['lag']<0, 'lag']/window ) / 4.
      else:
        # D2 - High DA
        #print(" D2 - High DA")
        nearest_collision.loc[nearest_collision['lag']>0, 'impact'] = 1. - nearest_collision.loc[nearest_collision['lag']>0, 'lag']/window
        nearest_collision.loc[nearest_collision['lag']<0, 'impact'] = - 1. - nearest_collision.loc[nearest_collision['lag']<0, 'lag']/window
    elif DA_level == 0:
      # Low DA (custom plasticity profiles)
      if aMSN % 2 == 0:
        # D1 - Low DA
        #print(" D1 - Low DA")
        nearest_collision.loc[nearest_collision['lag']>0, 'impact'] = - (1. - nearest_collision.loc[nearest_collision['lag']>0, 'lag']/window)
        nearest_collision.loc[nearest_collision['lag']<0, 'impact'] = (- 1. - nearest_collision.loc[nearest_collision['lag']<0, 'lag']/window )
      else:
        # D2 - Low DA
        #print(" D2 - Low DA")
        nearest_collision.loc[nearest_collision['lag']>0, 'impact'] = 1. - nearest_collision.loc[nearest_collision['lag']>0, 'lag']/window
        nearest_collision.loc[nearest_collision['lag']<0, 'impact'] = + 1. + nearest_collision.loc[nearest_collision['lag']<0, 'lag']/window
    change =  nearest_collision.groupby('CSNneurons')['impact'].sum()
    connections_to_update = nest.GetConnections(source=list(change.index), target=[aMSN])
    synapses_to_update = nest.GetStatus(connections_to_update, keys=['weight','source','receptor'])
    for syn_i in range(len(synapses_to_update)):
      syn = synapses_to_update[syn_i]
      if syn[2] == 1:
        neurot = base_weights['CSN_MSN']['AMPA']
      elif syn[2] == 2:
        neurot = base_weights['CSN_MSN']['NMDA']
      else:
        raise KeyError('synapse type must be 1 (AMPA) or 2 (NMDA)')
      new_w = syn[0] + change[syn[1]] * neurot # linear change
      # min_ratio and max_ratio define how much the weights can vary from the base weights
      if new_w > neurot * max_ratio:
        new_w = neurot * max_ratio
      elif new_w < neurot * min_ratio:
        new_w = neurot * min_ratio
      # viscosity is the maximal relative change that is allowed during one session (expressed relatively to the current weight, in %)
      if viscosity > 0.:
        max_new_w = syn[0] * (1. + viscosity/100.)
        min_new_w = syn[0] * (1. - viscosity/100.)
        if new_w > max_new_w:
          new_w = max_new_w
        elif new_w < min_new_w:
          new_w = min_new_w
      nest.SetStatus([connections_to_update[syn_i]], [{'weight': new_w}])

#-----------------------------------------------------------------------
def main():
  nest.set_verbosity("M_WARNING")
  
  print("Instantiate Test BG:")
  CSN_MSN = instantiate_BG(params, antagInjectionSite='none', antag='')
  get_strength(src=['CSN'], tgt=['MSN'])

  ## all changes as in the arm-reaching task
  #CSNFR=[2., 2. + 17.7 * 2.]
  #PActiveCSN=0.02824859 / 2.
  #PTNFR=[15., 15. + 31.3 * 2.]
  #PActivePTN=0.4792332 / 2.
  #CMPfFR=[4., 4. + 30. * 2.]
  #PActiveCMPf=0.05 / 2.

  # only the CSN changes from the arm-reaching task
  CSNFR=[2., 2. + 17.7 * 1.] # widespread
  PActiveCSN=0.02824859 / 1.
  PTNFR=[15., 15.]
  PActivePTN=0.
  CMPfFR=[4., 4.]
  PActiveCMPf=0.

  # reinforcement learning variables
  min_ratio = 0.5
  max_ratio = 1.5

  #min_ratio = 0.8 # more subtle
  #max_ratio = 1.2 # more subtle

  # `decrease`
  #min_ratio = 0.5 # try to lower MSN FR
  #max_ratio = 1.2 # by capping increase

  # `decreaseplus`
  #min_ratio = 0.5 # try to lower MSN FR
  #max_ratio = 1.1 # by capping increase

  # task structure
  #order = [0, 1] # good, then bad channel
  order = [1, 0] # bad, then good channel

  HighDA = 0 # positive reinforcement on channel A
  #HighDA = -1 # never do positive reinforcement

  #LowDA = 1 # negative reinforcement on channel B
  LowDA = -1 # never do negative reinforcement

  # first day: naive run
  for n in range(3):
    if n == 2:
      action = []
    else:
      action = [n]
    print("Naive run. Inputs on channels: "+str(action))
    checkAvgFRTst(params=params, CSNFR=CSNFR, PActiveCSN=PActiveCSN, PTNFR=PTNFR, PActivePTN=PActivePTN, CMPfFR=CMPfFR, PActiveCMPf=PActiveCMPf, action=action, antagInjectionSite='none', antag='')
    rotate_logs()

  # training
  trial = 1
  for days in range(3):
    for n in order:
      action = [n]
      print("Training run number "+str(trial)+". Inputs on channels: "+str(action))
      co_spikes = checkAvgFRTst(params=params, CSNFR=CSNFR, PActiveCSN=PActiveCSN, PTNFR=PTNFR, PActivePTN=PActivePTN, CMPfFR=CMPfFR, PActiveCMPf=PActiveCMPf, action=action, antagInjectionSite='none', antag='')
      rotate_logs()
      if n == HighDA:
        print("High DA plasticity")
        reinforcement_learning(co_spikes, CSN_MSN, DA_level = 1, min_ratio=min_ratio, max_ratio=max_ratio)
      elif n == LowDA:
        print("Low DA plasticity")
        reinforcement_learning(co_spikes, CSN_MSN, DA_level = 0, min_ratio=min_ratio, max_ratio=max_ratio)
      trial = trial + 1
      get_strength(src=['CSN'], tgt=['MSN'])

  # testing
  for n in range(3):
    if n == 2:
      action = []
    else:
      action = [n]
    print("Testing run. Inputs on channels: "+str(action))
    checkAvgFRTst(params=params, CSNFR=CSNFR, PActiveCSN=PActiveCSN, PTNFR=PTNFR, PActivePTN=PActivePTN, CMPfFR=CMPfFR, PActiveCMPf=PActiveCMPf, action=action, antagInjectionSite='none', antag='')
    rotate_logs()


#---------------------------
if __name__ == '__main__':
  main()
