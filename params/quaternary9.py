# Median parameters giving a plausibility score of 14/14 with all inDegree values set to 1/3

params = {'LG14modelID':9 ,
          'IeMSN':  24.5,
          'IeFSI':  8.  ,
          'IeSTN':  9.5 ,
          'IeGPe':  12. ,
          'IeGPi':  11. ,
          'nbMSN':  2644.*4, # original number of neurons, possibly scaled
          'nbFSI':  53.*4  , # ^
          'nbSTN':  8.*4   , # ^
          'nbGPe':  25.*4  , # ^
          'nbGPi':  14.*4  , # ^
          'nbCSN':  3000.*4, # large pool of CSN neurons (split per channel)
          'nbPTN':  3000.*4 , # large pool of PTN neurons (possibly split per channel)
          #'nbPTN':  100.*4 , # small pool of PTN neurons (possibly split per channel)
          'nbCMPf': 3000.*4, # large pool of thalamic neurons (not split per channel)
          'CSNFRmod':  [i/5. for i in range(6)],
          'PTNFRmod':  [i/5. for i in range(6)],
#          'CMPfFRmod': [i/5. for i in range(6)],
          'CMPfFRmod': [0., .4, 1.],
                  'cTypeCMPfMSN':   'focused',
                  'cTypeCMPfFSI':   'focused',
                  'cTypeCMPfSTN':   'focused',
                  'cTypeCMPfGPe':   'focused',
                  'cTypeCMPfGPi':   'focused',
          'simDuration': 5000., # long = 5000
          'selecRamp': True,
          }
