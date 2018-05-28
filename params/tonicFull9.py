# Hand-tuned parameters with custom inDegree values for original parameterization #9
# Degrees of freedom explored: only tonic inputs for all nuclei

params = {'LG14modelID': 9, # [0, 1, 2, 10, (12)] have ALPHA_GPe_MSN value != 0 
          'tSimu':       5000.,
                  
          'nestSeed':    3, # nest seed (affects input poisson spike trains)
          'pythonSeed':  3, # python seed (affects connection map)
          
          'IeMSN':              25., # 
          'IeFSI':              8., #
          'IeSTN':              9.2, #
          'IeGPe':              15.5, #
          'IeGPi':              13.5, #
          
          'nbMSN':              2644., # original number of neurons, possibly scaled
          'nbFSI':              53., # ^
          'nbSTN':              8., # ^
          'nbGPe':              25., # ^
          'nbGPi':              14., # ^
          'nbCSN':              3000., # large pool of CSN neurons (split per channel)
          'nbPTN':              100., # large pool of PTN neurons (possibly split per channel)
          'nbCMPf':             3000., # large pool of thalamic neurons (not split per channel)
          
          'inDegCMPfMSN':       0.33333, # inDegree from CMPf to * are 9
          'inDegCMPfFSI':       0.33333, # ^
          'inDegCMPfSTN':       0.33333, # ^
          'inDegCMPfGPe':       0.33333, # ^
          'inDegCMPfGPi':       0.33333, # ^
          'inDegPTNMSN':        0.33333, # inDegree from PTN to striatum are 3 (not a sensitive parameter)
          'inDegPTNFSI':        0.33333, # ^
          'inDegCSNMSN':        0.33333, # otherwise all other inDegrees are as in custom_noparrot_params9.py
          'inDegFSIMSN':        20.,
          'inDegMSNMSN':        0.33333,
          'inDegSTNMSN':        0.33333,
          'inDegGPeMSN':        0.33333,
          'inDegCSNFSI':        0.33333,
          'inDegSTNFSI':        0.33333,
          'inDegGPeFSI':        0.33333,
          'inDegFSIFSI':        0.33333,
          'inDegPTNSTN':        0.33333,
          'inDegGPeSTN':        0.33333,
          'inDegSTNGPe':        4,
          'inDegMSNGPe':        0.33333,
          'inDegGPeGPe':        0.33333,
          'inDegMSNGPi':        0.33333,
          'inDegSTNGPi':        0.33333,
          'inDegGPeGPi':        0.33333,
          
          'parrotCMPf': True,} # use parrot neurons for CMPf as well
          
#----- RESULTS -----
#* MSN - Rate: 0.296898638427 Hz -> OK (0.1 , 1)
#* FSI - Rate: 10.3811320755 Hz -> OK (5 , 20)
#* STN - Rate: 17.85 Hz -> OK (15.2 , 22.8)
#* GPe - Rate: 57.896 Hz -> OK (55.7 , 74.5)
#* GPi - Rate: 71.4 Hz -> OK (59.1 , 79.5)
#-------------------
#******************
# Score FRR: 5.0 / 5.0
#******************
#
# ------- PAUSES RESULTS --------------------
#>> /!\ NO !!! | No PAUSE DETECTED
#>> /!\ NO !!! | The longest ISI is:  67.1ms --> [250, ] Idealy 620ms
#
#***********************
# Score Pauses = 0.0 / 1.0
#***********************
#
#
#Activities at rest do not match: skipping deactivation tests
#******************
# Total Score : 5.0 / 6.0
#******************
