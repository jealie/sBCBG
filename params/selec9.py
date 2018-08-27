# Median parameters giving a plausibility score of 14/14 with all inDegree values set to 1/3

thescale = 4.
#thescale = 1.

params = {'LG14modelID':9 ,
          'IeMSN':  24.5,
          'IeFSI':  8.  ,
          'IeSTN':  9.5 ,
          'IeGPe':  12. ,
          'IeGPi':  11. ,
          'nbMSN':  2644.*thescale, # original number of neurons, possibly scaled
          'nbFSI':  53.*thescale  , # ^
          'nbSTN':  8.*thescale   , # ^
          'nbGPe':  25.*thescale  , # ^
          'nbGPi':  14.*thescale  , # ^
          'nbCSN':  3000.*thescale, # large pool of CSN neurons (split per channel)
          'nbPTN':  3000.*thescale , # large pool of PTN neurons (possibly split per channel)
          'nbCMPf': 3000.*thescale, # large pool of thalamic neurons (not split per channel)
          'cTypeCMPfMSN':   'focused',
          'cTypeCMPfFSI':   'focused',
          'cTypeCMPfSTN':   'focused',
          'cTypeCMPfGPe':   'focused',
          'cTypeCMPfGPi':   'focused',
          'selecRamp': True,
          }

