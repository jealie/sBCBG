# Median parameters giving a plausibility score of 14/14 with all inDegree values set to 1/3

the_scale = 1. # should be 4

params = {'LG14modelID':9 ,
          'offsetDuration': 500, # in ms
          'simDuration': 1000, # in ms
#          'nbCh': 4,
          'nbCh': 2,
          'nbCues': 2, # 2 cue channels
          'cTypeCSNMSN': 'focused', # learnt or not learnt?
          'IeMSN':  24.5, #24.5
          'IeFSI':  8.  ,
          'IeSTN':  9.5 ,
          'IeGPe':  12. ,
          'IeGPi':  11. ,
          'nbMSN':  2644.*the_scale, # original number of neurons, possibly scaled
          'nbFSI':  53.*the_scale  , # ^
          'nbSTN':  8.*the_scale   , # ^
          'nbGPe':  25.*the_scale  , # ^
          'nbGPi':  14.*the_scale  , # ^
          'nbCSN':  3000.*the_scale, # large pool of CSN neurons (split per channel)
          'nbPTN':  3000.*the_scale , # large pool of PTN neurons (possibly split per channel)
          'nbCMPf': 3000.*the_scale, # large pool of thalamic neurons (not split per channel)
          }
