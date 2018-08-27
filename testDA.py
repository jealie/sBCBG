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
# Initializes the network configuration (spike detectors, etc)
#------------------------------------------
def PrepareSim(params, PActiveCSN, PActivePTN, PActiveCMPf, recordPotential=False):
  #nest.ResetNetwork()
  #initNeurons()

  dataPath='log/'
  nest.SetKernelStatus({"overwrite_files":True}) # when we redo the simulation, we erase the previous traces

  nstrand.set_seed(params['nestSeed'], params['pythonSeed']) # sets the seed for the simulation

  # fetch custom values or set to default
  nbCh = params['nbCh']
  if 'offsetDuration' in params.keys():
    offsetDuration = float(params['offsetDuration'])
  else:
    offsetDuration = 500. # ms
  if 'simDuration' in params.keys():
    simDuration = float(params['simDuration'])
  else:
    simDuration = 1000. # ms
  if 'nbCues' in params.keys():
    nbCues = params['nbCues']
  else:
    nbCues = 0

  #-------------------------
  # prepare the lists of neurons that will be affected by the activity changes
  #-------------------------

  #ActPop = {'CSN':  [(),(),(),(),(),(),(),()],
  #          'PTN':  [(),(),(),(),(),(),(),()],
  #         'CMPf':  [(),(),(),(),(),(),(),()]}
  ActPop = {'CSN':  [(),(),(),()],
            'PTN':  [(),()],
           'CMPf':  [(),()]}
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
  activate_pop('CSN', PActiveCSN, nbCh+nbCues)
  activate_pop('PTN', PActivePTN, nbCh)
  activate_pop('CMPf', PActiveCMPf, nbCh)

  #-------------------------
  # measures
  #-------------------------
  spkDetect={} # spike detectors used to record the experiment
  multimeters={} # multimeters used to record one neuron in each population

  # single or multi-channel?
  if params['nbCh'] == 1:
    connect_detector = lambda N: nest.Connect(Pop[N], spkDetect[N])
    connect_multimeter = lambda N: nest.Connect(multimeters[N], [Pop[N][0]])
  else:
    connect_detector= lambda N: [nest.Connect(Pop[N][i], spkDetect[N]) for i in range(len(Pop[N]))]
    connect_multimeter = lambda N: nest.Connect(multimeters[N], [Pop[N][0][0]])

  for N in NUCLEI+['CSN', 'PTN', 'CMPf']:
    # 1000ms offset period for network stabilization
    spkDetect[N] = nest.Create("spike_detector", params={"withgid": True, "withtime": True, "label": N, "to_file": storeGDF})
    connect_detector(N)
    if recordPotential:
      # multimeter records only the last 200ms in one neuron in each population
      multimeters[N] = nest.Create('multimeter', params = {"withgid": True, 'withtime': True, 'interval': 0.1, 'record_from': ['V_m'], "label": N, "to_file": False})
      connect_multimeter(N)

  #-------------------------
  # packing
  #-------------------------
  sim_info = {'spkDetect': spkDetect, 
              'multimeters': multimeters,
              'ActPop': ActPop,
              'offsetDuration': offsetDuration,
              'simDuration': simDuration,
              'nbCh': nbCh,
              'nbCues': nbCues
             }
  return sim_info

#------------------------------------------
# Performs reinforcement learning in the BG
# CSNFR, PActiveCSN, PTNFR, PActivePTN, CMPfFR, PActiveCMPf: shape of BG inputs
# action: list of activated CSN channels
# cue: list of activated CSN cue channels
#------------------------------------------
def checkAvgFRTst(sim_info, CSNFR, PTNFR, CMPfFR, action=[], cue=[], showRasters=False, logFile = None):


  spkDetect = sim_info['spkDetect']
  multimeters = sim_info['multimeters']
  ActPop = sim_info['ActPop']
  offsetDuration = sim_info['offsetDuration']
  simDuration = sim_info['simDuration']
  nbCh = sim_info['nbCh']
  nbCues = sim_info['nbCues']

  showPotential = False # Switch to True to graph neurons' membrane potentials - does not handle well restarted simulations

  
  gCSN = CSNFR[1]-CSNFR[0]
  gPTN = PTNFR[1]-PTNFR[0]
  gCMPf = CMPfFR[1]-CMPfFR[0]
  maxChangeRel = np.zeros(nbCh)
  for a in action:
    maxChangeRel[a] = 1.
  for a in action:
    maxChangeRel[a] = 1.
  CSNrate =   gCSN * maxChangeRel
  PTNrate =   gPTN * maxChangeRel
  CMPfrate = gCMPf * maxChangeRel

  if nbCues > 0:
    CSNCuesrate = np.zeros(nbCues)
    for c in cue:
      CSNCuesrate[c] = gCSN
  else:
    print("/!\\ no cue configured in this circuit, will ignore /!\\")


  #-------------------------
  # Simulation
  #-------------------------

  # rest level
  for i in range(len(ActPop['CSN'])):
    nest.SetStatus(ActPop['CSN'][i], {'rate': CSNFR[0]})
  for i in range(len(ActPop['PTN'])):
    nest.SetStatus(ActPop['PTN'][i], {'rate': PTNFR[0]})
  for i in range(len(ActPop['CMPf'])):
    nest.SetStatus(ActPop['CMPf'][i], {'rate': CMPfFR[0]})
  # offset simulation
  if logFile != None:
    log(logFile, nest.GetKernelStatus('time'), nest.GetKernelStatus('time')+offsetDuration, [], [], None)
  nest.Simulate(offsetDuration)

  # reinforcement starts for the task
  simulationOffset = nest.GetKernelStatus('time')
  print('Simulation Offset: '+str(simulationOffset))

  # task inputs
  stepCSN = CSNFR[0] + CSNrate
  stepPTN = PTNFR[0] + PTNrate
  stepCMPf = CMPfFR[0] + CMPfrate
  for i in range(nbCh):
    stepCSN_i = max(0., stepCSN[i])
    stepPTN_i = max(0., stepPTN[i])
    stepCMPf_i = max(0., stepCMPf[i])
    print('Channel '+str(i)+':  CSN='+str(stepCSN_i)+'  PTN='+str(stepPTN_i)+'  CM/Pf='+str(stepCMPf_i))
    nest.SetStatus(ActPop['CSN'][i], {'rate': stepCSN_i})
    nest.SetStatus(ActPop['PTN'][i], {'rate': stepPTN_i})
    nest.SetStatus(ActPop['CMPf'][i], {'rate': stepCMPf_i})
  if nbCues > 0:
    stepCSN = CSNFR[0] + CSNCuesrate
    for i in range(nbCues):
      stepCSN_i = max(0., stepCSN[i])
      print('Cue '+str(i)+' (channel '+str(i+nbCh)+'):  CSN='+str(stepCSN_i))
      nest.SetStatus(ActPop['CSN'][i+nbCh], {'rate': stepCSN_i})
  else:
    print("/!\\ no cue configured in this circuit, will ignore /!\\")

  if logFile != None:
    log(logFile, nest.GetKernelStatus('time'), nest.GetKernelStatus('time')+simDuration, action, cue, None)
  nest.Simulate(simDuration)

  CSN_spikes = nest.GetStatus(spkDetect['CSN'], keys='events')[0]
  CSN_relevant = CSN_spikes['times'] > simulationOffset

  MSN_spikes = nest.GetStatus(spkDetect['MSN'], keys='events')[0]
  MSN_relevant = MSN_spikes['times'] > simulationOffset

  return [({'senders': MSN_spikes['senders'][MSN_relevant], 'times': MSN_spikes['times'][MSN_relevant]},),
          ({'senders': CSN_spikes['senders'][CSN_relevant], 'times': CSN_spikes['times'][CSN_relevant]},),
         ]

  #return [nest.GetStatus(spkDetect['MSN'], keys='events'), nest.GetStatus(spkDetect['CSN'], keys='events')]


def rotate_logs():
  import glob
  import shutil
  for logfile in glob.glob(r'log/*.gdf'):
    shutil.move(logfile, logfile+'.keep')

# ref_rela : against which value should the contrast be computed? False (default) <=> baseline, True <=> current mean
# to_rela : how should the contrast be enhanced? False (default) <=> weights set to extreme values relative to baseline, True <=> weights set relatively to their current values
def increase_contrast(base_weights, min_level = 0.5, max_level = 1.5, viscosity = 10, ref_rela=False, to_rela = False, CSNchannels=None, prob=None):
  from tqdm import tqdm
  if params['nbCh'] > 1 or ('nbCues' in params.keys() and params['nbCues'] > 1):
    unique_MSN = np.array(Pop['MSN']).flatten().tolist()
    if CSNchannels == None:
      CSN_IDs = np.array(Pop['CSN']).flatten().tolist()
    else:
      CSN_IDs = np.array(Pop['CSN'])[CSNchannels].flatten().tolist()
  else:
    CSN_IDs = Pop['CSN']
    unique_MSN = Pop['MSN']
  # contrast the weights
  AMPA_NMDA_keys = {0: 'AMPA', 1: 'NMDA'}
  for aMSN in tqdm(unique_MSN):
    connections_to_update = nest.GetConnections(source=CSN_IDs, target=[aMSN])
    current_synapses = pd.DataFrame(list(nest.GetStatus(connections_to_update, keys=['receptor','weight'])))
    if ref_rela == False:
      # select neurons to modify based on their distance to original baseline
      ref = [base_weights['CSN_MSN'][AMPA_NMDA_keys[0]], base_weights['CSN_MSN'][AMPA_NMDA_keys[1]]]
    else:
      # select neurons to modify based on their distance to current mean / median
      ref = current_synapses.groupby(0).mean()[1].tolist()
      #ref = current_synapses.groupby(0).median()[1].tolist()
    ri_idx = np.where(current_synapses[0] == 1)[0] # AMPA synapses
    maximize = np.where(current_synapses.loc[ri_idx, 1] > (1.+viscosity/100.)*ref[0])[0].tolist()
    minimize = np.where(current_synapses.loc[ri_idx, 1] < (1.-viscosity/100.)*ref[0])[0].tolist()
    if prob is not None:
      # select only a fraction of the synapses
      #import ipdb; ipdb.set_trace()
      if len(maximize):
        maximize = np.random.choice(maximize, size=int(np.ceil(len(maximize)*float(prob)))).tolist()
      if len(minimize):
        minimize = np.random.choice(minimize, size=int(np.ceil(len(minimize)*float(prob)))).tolist()
    for ri in [0, 1]:
      if to_rela == False:
        # set to the min/max values
        new_max = [max_level*base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]] for w in maximize]
        new_min = [min_level*base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]] for w in minimize]
      else:
        # adjust according to a ratio
        new_max = [max_level*current_synapses.loc[ri_idx[w], 1] for w in maximize]
        new_min = [min_level*current_synapses.loc[ri_idx[w], 1] for w in minimize]
      if len(maximize+minimize):
        nest.SetStatus((np.array(connections_to_update)[ri_idx+ri][maximize+minimize]).tolist(), [{'weight': new_weight} for new_weight in new_max+new_min])
      #ri_idx = np.where(current_synapses[0] == ri+1)[0]
      #maximize = np.where(current_synapses.loc[ri_idx, 1] > (1.+viscosity/100.)*ref[ri])[0].tolist()
      #minimize = np.where(current_synapses.loc[ri_idx, 1] < (1.-viscosity/100.)*ref[ri])[0].tolist()
      #if len(maximize) > 0:
      #  if to_rela == False:
      #    # set to the min/max values
      #    nest.SetStatus(np.array(connections_to_update)[ri_idx][maximize].tolist(), [{'weight': max_level*base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]]} for w in maximize])
      #  else:
      #    # adjust according to a ratio
      #    nest.SetStatus(np.array(connections_to_update)[ri_idx][maximize].tolist(), [{'weight': max_level*current_synapses.loc[ri_idx[w], 1]} for w in maximize])
      #if len(minimize) > 0:
      #  if to_rela == False:
      #    # set to the min/max values
      #    nest.SetStatus(np.array(connections_to_update)[ri_idx][minimize].tolist(), [{'weight': min_level*base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]]} for w in minimize])
      #  else:
      #    # adjust according to a ratio
      #    nest.SetStatus(np.array(connections_to_update)[ri_idx][minimize].tolist(), [{'weight': min_level*current_synapses.loc[ri_idx[w], 1]} for w in minimize])

# ref_rela : against which value should the contrast be computed? False (default) <=> baseline, True <=> current mean
# to_rela : how should the contrast be enhanced? False (default) <=> weights set to extreme values relative to baseline, True <=> weights set relatively to their current values
def trim_weakest(base_weights, prob, min_level = 0.1, viscosity = 10, ref_rela=False, CSNchannels=None):
  from tqdm import tqdm
  if params['nbCh'] > 1 or ('nbCues' in params.keys() and params['nbCues'] > 1):
    unique_MSN = np.array(Pop['MSN']).flatten().tolist()
    if CSNchannels == None:
      CSN_IDs = np.array(Pop['CSN']).flatten().tolist()
    else:
      CSN_IDs = np.array(Pop['CSN'])[CSNchannels].flatten().tolist()
  else:
    CSN_IDs = Pop['CSN']
    unique_MSN = Pop['MSN']
  # contrast the weights
  AMPA_NMDA_keys = {0: 'AMPA', 1: 'NMDA'}
  for aMSN in tqdm(unique_MSN):
    connections_to_update = nest.GetConnections(source=CSN_IDs, target=[aMSN])
    current_synapses = pd.DataFrame(list(nest.GetStatus(connections_to_update, keys=['receptor','weight'])))
    if ref_rela == False:
      # select neurons to modify based on their distance to original baseline
      ref = [base_weights['CSN_MSN'][AMPA_NMDA_keys[0]], base_weights['CSN_MSN'][AMPA_NMDA_keys[1]]]
    else:
      # select neurons to modify based on their distance to current mean / median
      ref = current_synapses.groupby(0).mean()[1].tolist()
      #ref = current_synapses.groupby(0).median()[1].tolist()
    ri_idx = np.where(current_synapses[0] == 1)[0] # AMPA synapses
    minimize = np.where(current_synapses.loc[ri_idx, 1] < (1.-viscosity/100.)*ref[0])[0].tolist()
    #import ipdb; ipdb.set_trace()
    if len(minimize):
      # select only the prob*N weakest synapses
      k = int(np.floor(len(minimize)*float(prob)))
      if k > 0:
        minimize = np.argpartition(minimize, k)[:k].tolist()
      else:
        return
    for ri in [0, 1]:
      # adjust according to a ratio
      new_min = [min_level*current_synapses.loc[ri_idx[w], 1] for w in minimize]
      nest.SetStatus((np.array(connections_to_update)[ri_idx+ri][minimize]).tolist(), [{'weight': new_weight} for new_weight in new_min])

def channel_contrast(base_weights, min_level = 0.5, max_level = 1.5, viscosity = 0, CSNchannels=None):
  from tqdm import tqdm
  if params['nbCh'] == 1:
    raise KeyError('Channel contrast is designed for the multi-channel case')
  if CSNchannels == None:
    CSNchannels = range(len(Pop['CSN']))
  for MSN_channel in range(len(Pop['MSN'])):
    MSN_IDs = np.array(Pop['MSN'])[MSN_channel].flatten().tolist()
    for CSN_channel in CSNchannels:
      CSN_IDs = np.array(Pop['CSN'])[CSN_channel].flatten().tolist()
      # contrast the weights
      AMPA_NMDA_keys = {0: 'AMPA', 1: 'NMDA'}
      connections_to_update = nest.GetConnections(source=CSN_IDs, target=MSN_IDs)
      current_synapses = pd.DataFrame(list(nest.GetStatus(connections_to_update, keys=['receptor','weight'])))
      current_AMPA_NMDA = current_synapses.groupby(0).mean()[1].tolist() # contrast wrt the mean
      #current_AMPA_NMDA = current_synapses.groupby(0).median()[1].tolist() # wrt to the median
      for ri in [0, 1]:
        ri_idx = np.where(current_synapses[0] == ri+1)[0]
        if current_AMPA_NMDA[ri] > (1.+viscosity/100.)*base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]]:
          nest.SetStatus(np.array(connections_to_update)[ri_idx].tolist(), [{'weight': max_level*base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]]} for w in range(len(ri_idx))])
          print('setting MSN['+str(MSN_channel)+'] -> CSN['+str(CSN_channel)+'] to max value')
        elif current_AMPA_NMDA[ri] < (1.-viscosity/100.)*base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]]:
          nest.SetStatus(np.array(connections_to_update)[ri_idx].tolist(), [{'weight': min_level*base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]]} for w in range(len(ri_idx))])
          print('setting MSN['+str(MSN_channel)+'] -> CSN['+str(CSN_channel)+'] to min value')
        else:
          print('MSN['+str(MSN_channel)+'] -> CSN['+str(CSN_channel)+'] is within '+str(viscosity)+'% of baseline, not touching')

def renorm_mean_baseline(base_weights, CSNchannels=None, MSNs = None):
  from tqdm import tqdm
  if params['nbCh'] > 1 or ('nbCues' in params.keys() and params['nbCues'] > 1):
    if MSNs is None:
      unique_MSN = np.array(Pop['MSN']).flatten().tolist()
    if CSNchannels is None:
      CSN_IDs = np.array(Pop['CSN']).flatten().tolist()
    else:
      CSN_IDs = np.array(Pop['CSN'])[CSNchannels].flatten().tolist()
  else:
    CSN_IDs = Pop['CSN']
    if MSNs is None:
      unique_MSN = Pop['MSN']
  if MSNs is not None:
    # normalize only pre-defined MSN
    unique_MSN = MSNs
  # contrast the weights
  AMPA_NMDA_keys = {0: 'AMPA', 1: 'NMDA'}
  for aMSN in tqdm(unique_MSN):
    connections_to_update = nest.GetConnections(source=CSN_IDs, target=[aMSN])
    current_synapses = pd.DataFrame(list(nest.GetStatus(connections_to_update, keys=['receptor','weight'])))
    current_AMPA_NMDA = current_synapses.groupby(0).mean()[1].tolist() # adjust the mean
    #current_AMPA_NMDA = current_synapses.groupby(0).median()[1].tolist() # adjust the median
    for ri in [0, 1]:
      ratio = current_AMPA_NMDA[ri] / base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]]
      ri_idx = np.where(current_synapses[0] == ri+1)[0]
      nest.SetStatus(np.array(connections_to_update)[ri_idx].tolist(), [{'weight': w} for w in (current_synapses.loc[ri_idx, 1] / ratio)])

# DA modeling according to a linear function (for debug)
def linear_impact(nearest_collision, DA_level, aMSN, window=50.):
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
  return nearest_collision

# DA modeling according to a linear function (for debug)
def stdp_impact(nearest_collision, DA_level, aMSN, tau=10., A=0.05):
  if DA_level == 1:
    # High DA (mostly classical hebbian)
    if aMSN % 2 == 0:
      # D1 - High DA
      #print("D1 - High DA")
      nearest_collision.loc[nearest_collision['lag']>0, 'impact'] = A * np.exp(np.array(nearest_collision.loc[nearest_collision['lag']>0, 'lag']) / tau)
      nearest_collision.loc[nearest_collision['lag']<0, 'impact'] = -A * 0.25 * np.exp(np.array(-nearest_collision.loc[nearest_collision['lag']<0, 'lag']) / tau)
    else:
      # D2 - High DA
      #print(" D2 - High DA")
      nearest_collision.loc[nearest_collision['lag']>0, 'impact'] = A * np.exp(np.array(nearest_collision.loc[nearest_collision['lag']>0, 'lag']) / tau)
      nearest_collision.loc[nearest_collision['lag']<0, 'impact'] = -A * np.exp(np.array(-nearest_collision.loc[nearest_collision['lag']<0, 'lag']) / tau)
  elif DA_level == 0:
    # Low DA (custom plasticity profiles)
    if aMSN % 2 == 0:
      # D1 - Low DA
      #print(" D1 - Low DA")
      import ipdb; ipdb.set_trace()
      #nearest_collision.loc[nearest_collision['lag']>0, 'impact'] = - (1. - nearest_collision.loc[nearest_collision['lag']>0, 'lag']/window)
      #nearest_collision.loc[nearest_collision['lag']<0, 'impact'] = (- 1. - nearest_collision.loc[nearest_collision['lag']<0, 'lag']/window )
    else:
      # D2 - Low DA
      #print(" D2 - Low DA")
      import ipdb; ipdb.set_trace()
      #nearest_collision.loc[nearest_collision['lag']>0, 'impact'] = 1. - nearest_collision.loc[nearest_collision['lag']>0, 'lag']/window
      #nearest_collision.loc[nearest_collision['lag']<0, 'impact'] = + 1. + nearest_collision.loc[nearest_collision['lag']<0, 'lag']/window
  return nearest_collision

def reinforcement_learning(co_spikes, base_weights, DA_level, min_ratio = 0.5, max_ratio = 1.5, viscosity = 0., CSNchannels=None, window = 50.):
  from tqdm import tqdm
  unique_MSN = np.unique(co_spikes[0][0]['senders'])
  MSN_spikes = pd.DataFrame({'MSNtimes': co_spikes[0][0]['times'], 'MSNneurons': co_spikes[0][0]['senders']})
  MSN_spikes.set_index('MSNtimes', inplace=True, drop=False)
  CSN_spikes = pd.DataFrame({'CSNtimes': co_spikes[1][0]['times'], 'CSNneurons': co_spikes[1][0]['senders']})
  CSN_spikes.set_index('CSNtimes', inplace=True, drop=False)
  if params['nbCh'] > 1 or ('nbCues' in params.keys() and params['nbCues'] > 1):
    if CSNchannels == None:
      CSN_IDs = np.array(Pop['CSN']).flatten().tolist()
    else:
      CSN_IDs = np.array(Pop['CSN'])[CSNchannels].flatten().tolist()
  else:
    CSN_IDs = Pop['CSN']
  all_targetingCSN = pd.DataFrame.from_records(nest.GetConnections(source=CSN_IDs, target=unique_MSN.tolist()))
  all_targetingCSN.columns = ['tgt','src','c','d','e']
  for aMSN in tqdm(unique_MSN):
    targetingCSN = all_targetingCSN.loc[all_targetingCSN['src']==aMSN, 'tgt']
    CSN_spiking_times = CSN_spikes.loc[CSN_spikes['CSNneurons'].isin(targetingCSN)]
    MSN_spiking_times = MSN_spikes.loc[MSN_spikes['MSNneurons'] == aMSN]
    nearest_collision = MSN_spiking_times.reindex(CSN_spiking_times.index, method='nearest')
    nearest_collision['lag'] = nearest_collision['MSNtimes'] - nearest_collision.index # time between CSN and MSN firing (negative lag <=> CSN fired after MSN)
    nearest_collision = nearest_collision.loc[nearest_collision['lag'].abs() < window] # keep only close collisions
    nearest_collision = nearest_collision.merge(CSN_spiking_times[['CSNneurons']], how='left', left_index=True, right_index=True) # merge back the CSN neuron ID
    # classical hebbian learning (post after pre: strengthening, post before pre: weakening)
    # or something else?
    #nearest_collision = linear_impact(nearest_collision, DA_level, aMSN, window=window)
    nearest_collision = stdp_impact(nearest_collision, DA_level, aMSN)
    #change =  nearest_collision.groupby('CSNneurons')['impact'].mean()
    change =  nearest_collision.groupby('CSNneurons')['impact'].sum()
    try:
      connections_to_update = nest.GetConnections(source=list(change.index), target=[aMSN])
      synapses_to_update = nest.GetStatus(connections_to_update, keys=['weight','source','receptor'])
      all_new_w = []
      for syn_i in range(len(synapses_to_update)):
        syn = synapses_to_update[syn_i]
        if syn[2] == 1:
          neurot = base_weights['CSN_MSN']['AMPA']
        elif syn[2] == 2:
          neurot = base_weights['CSN_MSN']['NMDA']
        else:
          raise KeyError('synapse type must be 1 (AMPA) or 2 (NMDA)')
        new_w = syn[0] + change[syn[1]] * neurot # additive change
        ### viscosity is the maximal relative change that is allowed during one session (expressed relatively to the current weight, in %)
        ##if viscosity > 0.:
        ##  max_new_w = syn[0] * (1. + viscosity/100.)
        ##  min_new_w = syn[0] * (1. - viscosity/100.)
        ##  if new_w > max_new_w:
        ##    new_w = max_new_w
        ##  elif new_w < min_new_w:
        ##    new_w = min_new_w
        # viscosity discounts changes (expressed relatively to the current weight, in %)
        if viscosity > 0.:
          #print("original: "+str(syn[0]))
          #print("before cap: "+str(new_w))
          new_w = syn[0] + change[syn[1]] * (viscosity / 100.) * neurot # linear change
          #print("after cap: "+str(new_w))
        # min_ratio and max_ratio define how much the weights can vary from base value
        if new_w > neurot * max_ratio:
          new_w = neurot * max_ratio
        elif new_w < neurot * min_ratio:
          new_w = neurot * min_ratio
        #nest.SetStatus([connections_to_update[syn_i]], [{'weight': new_w}])
        all_new_w += [{'weight': new_w}]
      nest.SetStatus(connections_to_update, all_new_w)
    except:
      print("no connection to update for neuron "+str(aMSN)+"!\n")
  return unique_MSN # using in a (potential) second pass, to normalize the weights
  #AMPA_NMDA_keys = {0: 'AMPA', 1: 'NMDA'}
  #for aMSN in tqdm(unique_MSN):
  #  connections_to_update = nest.GetConnections(source=CSN_IDs, target=[aMSN])
  #  current_synapses = pd.DataFrame(list(nest.GetStatus(connections_to_update, keys=['receptor','weight'])))
  #  current_AMPA_NMDA = current_synapses.groupby(0).mean()[1].tolist() # adjust the mean
  #  #current_AMPA_NMDA = current_synapses.groupby(0).median()[1].tolist() # adjust the median
  #  for ri in [0, 1]:
  #    ratio = current_AMPA_NMDA[ri] / base_weights['CSN_MSN'][AMPA_NMDA_keys[ri]]
  #    ri_idx = np.where(current_synapses[0] == ri+1)[0]
  #    nest.SetStatus(np.array(connections_to_update)[ri_idx].tolist(), [{'weight': w} for w in (current_synapses.loc[ri_idx, 1] / ratio)])

def log(logFile, Tstart, Tend, actions, cues, DA):
  logFile.write(str(Tstart)+', '+str(Tend))
  for n in range(params['nbCh']):
    if actions == None:
      val_isin = 'NA'
    else:
      val_isin = (n in actions)
    logFile.write(', '+str(val_isin))
  for n in range(params['nbCues']):
    if cues == None:
      val_isin = 'NA'
    else:
      val_isin = (n in cues)
    logFile.write(', '+str(val_isin))
  if DA == None:
    logFile.write(', NA\n')
  else:
    logFile.write(', '+str(DA)+'\n')

def log_weights(logFileW, T, CSNchannels=None):
  if params['nbCh'] == 1:
    raise KeyError('Weight logging is designed for the multi-channel case')
  if CSNchannels == None:
    CSNchannels = range(len(Pop['CSN']))
  for MSN_channel in range(len(Pop['MSN'])):
    MSN_IDs = np.array(Pop['MSN'])[MSN_channel].flatten().tolist()
    for CSN_channel in CSNchannels:
      #lbl = 'CSN['+str(CSN_channel)+'] -> MSN['+str(MSN_channel)+']'
      lbl = 'C'+str(CSN_channel)+'M'+str(MSN_channel)
      CSN_IDs = np.array(Pop['CSN'])[CSN_channel].flatten().tolist()
      AMPA_NMDA_keys = {0: 'AMPA', 1: 'NMDA'}
      connections_to_update = nest.GetConnections(source=CSN_IDs, target=MSN_IDs)
      current_synapses = pd.DataFrame(list(nest.GetStatus(connections_to_update, keys=['receptor','weight','source','target'])))
      current_synapses['lbl'] = lbl
      current_synapses['T'] = T
      current_synapses.to_csv(logFileW, header=False, index=False)

## Simplistic experiment - create only required nuclei and connect them
def createMini(params):
  nest.ResetKernel()
  dataPath='log/'
  if 'nbcpu' in params:
    nest.SetKernelStatus({'local_num_threads': params['nbcpu']})

  nstrand.set_seed(params['nestSeed'], params['pythonSeed']) # sets the seed for the BG construction

  nest.SetKernelStatus({"data_path": dataPath})
  #nest.SetKernelStatus({"resolution": 0.005}) # simulates with a higher precision
  initNeurons()

  print '/!\ Using the following LG14 parameterization',params['LG14modelID']
  loadLG14params(params['LG14modelID'])
  loadThetaFromCustomparams(params)

  def create_pop(*args, **kwargs):
    if 'nbCh' not in kwargs.keys():
      # enforce the default
      kwargs['nbCh'] = params['nbCh']
    createMC(*args, **kwargs)
  update_Ie = lambda p: [nest.SetStatus(Pop[p][i],{"I_e":params['Ie'+p]}) for i in range(len(Pop[p]))]

  nbSim['MSN'] = params['nbMSN']
  create_pop('MSN')
  update_Ie('MSN')

  nbSim['CSN'] = params['nbCSN']
  if 'nbCues' in params.keys():
    # cue channels are present
    CSNchannels = params['nbCh']+params['nbCues']
  else:
    CSNchannels = params['nbCh']
  create_pop('CSN', nbCh=CSNchannels, fake=True, parrot=True)

  nbSim['PTN'] = params['nbPTN']
  create_pop('PTN', fake=True, parrot=True)

  nbSim['CMPf'] = params['nbCMPf']
  create_pop('CMPf', fake=True, parrot=True)

  # fake populations
  rate['FSI'] = 10.
  nbSim['FSI'] = params['nbFSI']
  create_pop('FSI', fake=True, parrot=True)

  rate['STN'] = 20.
  nbSim['STN'] = params['nbSTN']
  create_pop('STN')

  rate['GPe'] = 75.
  nbSim['GPe'] = params['nbGPe']
  create_pop('GPe')

  rate['GPi'] = 80.
  nbSim['GPi'] = params['nbGPi']
  create_pop('GPi')

  def connect_pop(*args, **kwargs):
    if 'source_channels' not in kwargs.keys():
      # enforce the default
      kwargs['source_channels'] = range(params['nbCh'])
    return connectMC(*args, RedundancyType=params['RedundancyType'], stochastic_delays=params['stochastic_delays'], verbose=True, **kwargs)

  #-------------------------
  # connection of populations
  #-------------------------
  G = {'MSN': params['GMSN'],
       'FSI': params['GFSI'],
       'STN': params['GSTN'],
       'GPe': params['GGPe'],
       'GPi': params['GGPi'],
      }
  print '\nConnecting neurons\n================'
  print '* MSN Inputs'
  if 'nbCues' not in params.keys():
    # usual case: CSN have as the same number of channels than the BG nuclei
    CSN_MSN = connect_pop('ex','CSN','MSN', projType=params['cTypeCSNMSN'], redundancy=params['redundancyCSNMSN'], gain=G['MSN'])
  else:
    # special case: extra 'cue' channels that target MSN
    CSN_MSN = connect_pop('ex','CSN','MSN', projType=params['cTypeCSNMSN'], redundancy=params['redundancyCSNMSN'], gain=G['MSN']/2., source_channels=range(params['nbCh']))
    connect_pop('ex','CSN','MSN', projType='diffuse', redundancy=params['redundancyCSNMSN'], gain=G['MSN']/2., source_channels=range(params['nbCh'], params['nbCh']+params['nbCues']))
  PTN_MSN = connect_pop('ex','PTN','MSN', projType=params['cTypePTNMSN'], redundancy= params['redundancyPTNMSN'], gain=G['MSN'])
  CMPf_MSN = connect_pop('ex','CMPf','MSN',projType=params['cTypeCMPfMSN'],redundancy= params['redundancyCMPfMSN'],gain=G['MSN'])
  connect_pop('in','MSN','MSN', projType=params['cTypeMSNMSN'], redundancy= params['redundancyMSNMSN'], gain=G['MSN'])
  connect_pop('in','FSI','MSN', projType=params['cTypeFSIMSN'], redundancy= params['redundancyFSIMSN'], gain=G['MSN'])

  base_weights = {'CSN_MSN': CSN_MSN, 'PTN_MSN': PTN_MSN, 'CMPf_MSN': CMPf_MSN}

  return base_weights

def plotCSNMSNstrength(CSN_MSN):
  M11 = get_strength(src=['CSN'], tgt=['MSN'], srcCh=0, tgtCh=0, verbose=False)
  M22 = get_strength(src=['CSN'], tgt=['MSN'], srcCh=1, tgtCh=1, verbose=False)
  C11 = get_strength(src=['CSN'], tgt=['MSN'], srcCh=2, tgtCh=0, verbose=False)
  C12 = get_strength(src=['CSN'], tgt=['MSN'], srcCh=2, tgtCh=1, verbose=False)
  C21 = get_strength(src=['CSN'], tgt=['MSN'], srcCh=3, tgtCh=0, verbose=False)
  C22 = get_strength(src=['CSN'], tgt=['MSN'], srcCh=3, tgtCh=1, verbose=False)
  for rec in ['AMPA', 'NMDA']:
    print('Receptor '+rec+':')
    print('M11: '+str(M11[0]['weight']['CSN']['MSN'][rec] / CSN_MSN['CSN_MSN'][rec]))
    print(str(M11[1]['weight']['CSN']['MSN'][rec] / [CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec]]))
    print('M22: '+str(M22[0]['weight']['CSN']['MSN'][rec] / CSN_MSN['CSN_MSN'][rec]))
    print(str(M22[1]['weight']['CSN']['MSN'][rec] / [CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec]]))
    print('C11: '+str(C11[0]['weight']['CSN']['MSN'][rec] / CSN_MSN['CSN_MSN'][rec]))
    print(str(C11[1]['weight']['CSN']['MSN'][rec] / [CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec]]))
    print('C12: '+str(C12[0]['weight']['CSN']['MSN'][rec] / CSN_MSN['CSN_MSN'][rec]))
    print(str(C12[1]['weight']['CSN']['MSN'][rec] / [CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec]]))
    print('C21: '+str(C21[0]['weight']['CSN']['MSN'][rec] / CSN_MSN['CSN_MSN'][rec]))
    print(str(C21[1]['weight']['CSN']['MSN'][rec] / [CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec]]))
    print('C22: '+str(C22[0]['weight']['CSN']['MSN'][rec] / CSN_MSN['CSN_MSN'][rec]))
    print(str(C22[1]['weight']['CSN']['MSN'][rec] / [CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec], CSN_MSN['CSN_MSN'][rec]]))

#-----------------------------------------------------------------------
def main2():
  nest.set_verbosity("M_WARNING")
  
  #########################
  # SIMULATION PARAMETERS #
  #########################

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
  min_ratio = 0.5 # first try
  max_ratio = 1.5

  #min_ratio = 0.5 # higher reinforcement
  #max_ratio = 2.

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

  #reinforced_channels = range(params['nbCh'], params['nbCh']+params['nbCues']) # only Cues
  reinforced_channels = range(params['nbCh']+params['nbCues']) # Actions & Cues


  # BG instantiation
  print("Instantiate Test BG:")
  CSN_MSN = instantiate_BG(params, antagInjectionSite='none', antag='')
  sim_info = PrepareSim(params, PActiveCSN=PActiveCSN, PActivePTN=PActivePTN, PActiveCMPf=PActiveCMPf)
  #get_strength(src=['CSN'], tgt=['MSN'])

  logXP = open('log.csv','w', 0)
  logXP.write('Tstart, Tend')
  for n in range(params['nbCh']):
    logXP.write(', Action_'+str(n))
  for n in range(params['nbCues']):
    logXP.write(', Cue_'+str(n))
  logXP.write(', DA\n')

  ## first day: naive run
  #for cue_i in [0,1,2]:
  #  if cue_i == 2:
  #    cue = []
  #  else:
  #    cue = [cue_i]
  #  for action_i in [0,1,2]:
  #    if action_i == 2:
  #      action = []
  #    else:
  #      action = [action_i]
  #    print("Naive run. Inputs on channels: "+str(action)+"  Cues on channels: "+str(cue))
  #    checkAvgFRTst(sim_info=sim_info, CSNFR=CSNFR, PTNFR=PTNFR, CMPfFR=CMPfFR, action=action, cue=cue, logFile = logXP)
  #    #rotate_logs()

  # training
  trial = 1
  for days in range(3):
    for cue_i in [0,1,2]:
      if cue_i == 2:
        cue = []
      else:
        cue = [cue_i]
      for n in order:
        action = [n]
        DA = None
        if n == HighDA and cue_i == 0:
          print("Good channel selected in Good context => High DA plasticity")
          DA = 1
        elif n == HighDA and cue_i == 1:
          print("Good channel selected in Wrong context => Low DA plasticity")
          DA = 0
        elif n == LowDA:
          print("Wrong channel selected => Low DA plasticity")
          DA = 0
        ###### No point in trying non-reinforced trials ->
        #####print("Training run number "+str(trial)+". Inputs on channels: "+str(action))
        #####co_spikes = checkAvgFRTst(sim_info=sim_info, CSNFR=CSNFR, PTNFR=PTNFR, CMPfFR=CMPfFR, action=action, cue=cue, logFile = logXP)
        ######rotate_logs()
        if DA != None:
          co_spikes = checkAvgFRTst(sim_info=sim_info, CSNFR=CSNFR, PTNFR=PTNFR, CMPfFR=CMPfFR, action=action, cue=cue, logFile = logXP)
          if DA == 1:
            viscosity = 0. # unlimited plasticity
          else:
            viscosity = 50. # weights can not change from more than x%
          log(logXP, nest.GetKernelStatus('time')-sim_info['simDuration'], nest.GetKernelStatus('time'), None, None, DA)
          reinforcement_learning(co_spikes, CSN_MSN, DA_level = DA, min_ratio=min_ratio, max_ratio=max_ratio, CSNchannels = reinforced_channels, viscosity = viscosity)
        trial = trial + 1
        #get_strength(src=['CSN'], tgt=['MSN'])
    for cue_i in [0,1]:
      cue = [cue_i]
      action = []
      print("Daily test. Inputs on channels "+str(action)+" with cues "+str(cue))
      checkAvgFRTst(sim_info=sim_info, CSNFR=CSNFR, PTNFR=PTNFR, CMPfFR=CMPfFR, action=action, cue=cue, logFile = logXP)

  # testing
  #for n in [2,0,1]:
  for n in [0,1]: # no action was already tested before
    for cue_i in [2,0,1]:
      if cue_i == 2:
        cue = []
      else:
        cue = [cue_i]
      if n == 2:
        action = []
      else:
        action = [n]
      print("Testing run. Inputs on channels "+str(action)+" with cues "+str(cue))
      checkAvgFRTst(sim_info=sim_info, CSNFR=CSNFR, PTNFR=PTNFR, CMPfFR=CMPfFR, action=action, cue=cue, logFile = logXP)
      #rotate_logs()
  
  logXP.close()


#-----------------------------------------------------------------------
def main():
  nest.set_verbosity("M_WARNING")
  
  #########################
  # SIMULATION PARAMETERS #
  #########################

  ## all changes as in the arm-reaching task
  #CSNFR=[2., 2. + 17.7 * 2.]
  #PActiveCSN=0.02824859 / 2.
  #PTNFR=[15., 15. + 31.3 * 2.]
  #PActivePTN=0.4792332 / 2.
  #CMPfFR=[4., 4. + 30. * 2.]
  #PActiveCMPf=0.05 / 2.

  # only the CSN changes from the arm-reaching task
  #spread_factor = 2. # default, fits visually with Georgopoulos 1982
  #spread_factor = 1. # somewhat more widespread than the original value of 2
  #spread_factor = 0.1 # much more widespread than the original value of 2
  spread_factor = 0.02824859 # maximum spread
  CSNFR=[2., 2. + 17.7 * spread_factor]
  PActiveCSN=0.02824859 / spread_factor
  PTNFR=[15., 15.]
  PActivePTN=0.
  CMPfFR=[4., 4.]
  PActiveCMPf=0.

  # reinforcement learning variables
  #min_ratio = 0.5 # first try
  #max_ratio = 1.5

  #min_ratio = 0.1 # lower min reinforcement
  #max_ratio = 1.5

  #min_ratio = 0.5 # higher reinforcement
  #max_ratio = 2.

  #min_ratio = 0.8 # more subtle
  #max_ratio = 1.2 # more subtle

  #min_ratio = 0. # basically unlimited
  #max_ratio = 3.

  #min_ratio = 0. # activated synapses compensate de-activated ones
  #max_ratio = 2.

  min_ratio = 0.1 # activated synapses compensate de-activated ones
  max_ratio = 1.9 # no synapse can go completely extinct

  ## {cue: {action: reward}}
  #rewards = {0: {0: 1, 1: None},    # Rewards only Cue 1 - Action 1
  #           1: {0: None, 1: None}} #
  rewards = {0: {0: 1, 1: None},    # Rewards Cue 1 - Action 1
             1: {0: None, 1: 1}}    # & Cue 2 - Action 2
  #rewards = {0: {0: 1, 1: None},
  #           1: {0: None, 1: 1}}
  #rewards = {0: {0: 1, 1: 0},
  #           1: {0: 0, 1: 1}}

  reinforced_channels = range(params['nbCh'], params['nbCh']+params['nbCues']) # only Cues
  #reinforced_channels = range(params['nbCh']+params['nbCues']) # Actions & Cues

  # BG instantiation
  print("Instantiate Test BG:")
  CSN_MSN = instantiate_BG(params, antagInjectionSite='none', antag='')
  #CSN_MSN = createMini(params)

  sim_info = PrepareSim(params, PActiveCSN=PActiveCSN, PActivePTN=PActivePTN, PActiveCMPf=PActiveCMPf)
  #get_strength(src=['CSN'], tgt=['MSN'])

  logXP = open('log.csv','w', 0)
  logXP.write('Tstart, Tend')
  for n in range(params['nbCh']):
    logXP.write(', Action_'+str(n))
  for n in range(params['nbCues']):
    logXP.write(', Cue_'+str(n))
  logXP.write(', DA\n')

  logW = open('logW.csv', 'w', 0)
  logW.write('rec,W,source,target,label,Time\n')

  # training: actions are forced, reinf makes the association
  trial = 1
  for days in range(100):
    for cue_i in np.random.permutation([0,1]).tolist():
      if cue_i == 2:
        cue = []
      else:
        cue = [cue_i]
      for n in np.random.permutation([0,1]).tolist():
        action = [n]
        print("Training run number "+str(trial)+" within session "+str(days)+". Inputs on action "+str(action)+" with cues "+str(cue))
        DA = None
        if rewards[cue_i][n] == 1:
          print("Good channel selected in Good context => High DA plasticity")
          DA = 1
        elif rewards[cue_i][n] == 0:
          print("Wrong channel or context => Low DA plasticity")
          DA = 0
        ###### No point in trying non-reinforced trials ->
        #####print("Training run number "+str(trial)+". Inputs on channels: "+str(action))
        #####co_spikes = checkAvgFRTst(sim_info=sim_info, CSNFR=CSNFR, PTNFR=PTNFR, CMPfFR=CMPfFR, action=action, cue=cue, logFile = logXP)
        ######rotate_logs()
        if DA != None:
          co_spikes = checkAvgFRTst(sim_info=sim_info, CSNFR=CSNFR, PTNFR=PTNFR, CMPfFR=CMPfFR, action=action, cue=cue, logFile = logXP)
          #viscosity = 10.
          viscosity = 0. # required to enable multiplicative changes
          #viscosity = 300. # larger changes
          #if DA == 1:
          #  viscosity = 0. # unlimited plasticity
          #else:
          #  viscosity = 0. # weights can not change from more than x%
          log(logXP, nest.GetKernelStatus('time')-sim_info['simDuration'], nest.GetKernelStatus('time'), None, None, DA)
          MSNs = reinforcement_learning(co_spikes, CSN_MSN, DA_level = DA, min_ratio=min_ratio, max_ratio=max_ratio, CSNchannels = reinforced_channels, viscosity = viscosity)
          trim_weakest(CSN_MSN, 0.05, min_level = 0.1, viscosity = 1, ref_rela=True, CSNchannels=reinforced_channels)
          renorm_mean_baseline(CSN_MSN, CSNchannels = reinforced_channels, MSNs = MSNs)
          #increase_contrast(CSN_MSN, min_level=0.9, max_level = 1.1, viscosity=1, ref_rela=False, to_rela = True, CSNchannels = reinforced_channels)
          ##increase_contrast(CSN_MSN, min_level=0.5, max_level = 1.5, viscosity=0, ref_rela=False, to_rela = True, CSNchannels = reinforced_channels, prob=0.2)
          #renorm_mean_baseline(CSN_MSN, CSNchannels = reinforced_channels)
          log_weights(logW, nest.GetKernelStatus('time'), CSNchannels = reinforced_channels)
          # Some debug info
          #plotCSNMSNstrength(CSN_MSN)
        trial = trial + 1
        #get_strength(src=['CSN'], tgt=['MSN'])
    #if days % 5 == 4:
    if True:
      for cue_i in [0,1]:
        cue = [cue_i]
        action = []
        print("Test. Inputs on channels "+str(action)+" with cues "+str(cue))
        checkAvgFRTst(sim_info=sim_info, CSNFR=CSNFR, PTNFR=PTNFR, CMPfFR=CMPfFR, action=action, cue=cue, logFile = logXP)

  print('BEFORE CONTRASTING ACTIVITIES')
  plotCSNMSNstrength(CSN_MSN)
  #increase_contrast(CSN_MSN, min_level=0.1, max_level = 2.9, viscosity=10, CSNchannels = reinforced_channels)
  #increase_contrast(CSN_MSN, min_level=0.1, max_level = 2.9, viscosity=1, CSNchannels = range(params['nbCh'], params['nbCh']+params['nbCues'])) # only Cues
  channel_contrast(CSN_MSN, min_level=0.1, max_level = 1.9, viscosity=0.1, CSNchannels = range(params['nbCh'], params['nbCh']+params['nbCues'])) # only Cues

  print('AFTER CONTRASTING ACTIVITIES')
  plotCSNMSNstrength(CSN_MSN)

  # free roaming
  #for n in [2,0,1]:
  #for n in [0,1]: # no action was already tested before
  for n in [2]: # no action
    for cue_i in [2,0,1]:
      if cue_i == 2:
        cue = []
      else:
        cue = [cue_i]
      if n == 2:
        action = []
      else:
        action = [n]
      print("Testing run. Inputs on channels "+str(action)+" with cues "+str(cue))
      checkAvgFRTst(sim_info=sim_info, CSNFR=CSNFR, PTNFR=PTNFR, CMPfFR=CMPfFR, action=action, cue=cue, logFile = logXP)
      #rotate_logs()
  
  logXP.close()



#---------------------------
if __name__ == '__main__':
  main()
