
params = {'LG14modelID': 9, # [0, 1, 2, 10, (12)] have ALPHA_GPe_MSN value != 0 
          'tSimu':       20000.,
                  
          'nestSeed':    10, # nest seed (affects input poisson spike trains)
          'pythonSeed':  10, # python seed (affects connection map)
###############################################################################          
          'nbMSN':       2644.,
          'nbFSI':       53.,
          'nbSTN':       8.,
          'nbGTI':       20., # 84% GPe --> 80% GPe ( SATO 2000 )
          'nbGTA':       5., # 16% GPe --> 20% GPe ( SATO 2000 )
          'nbGPi':       14.,
          'nbCSN':       3000.,
          'nbPTN':       100.,
          'nbCMPf':      100.,
###############################################################################         
          'IeMSN':       25., # 
          'IeFSI':       8., #
          'IeSTN':       9., 
          'IeGTA':       15.5, # 15.5, #
          'IeGTI':       15.5, # 15.5, #
          'IeGPi':       13.5, #
###############################################################################         
          'gainCSNMSN': 1.,
          'gainCSNFSI': 1.,
                  
          'gainPTNMSN': 1.,
          'gainPTNFSI': 1.,
          'gainPTNSTN': 1.,
                
          'gainCMPfMSN':1.,
          'gainCMPfFSI':1.,
          'gainCMPfSTN':1.,
          'gainCMPfGTA':1.,
          'gainCMPfGTI':1.,
          'gainCMPfGPi':1.,
                 
          'gainMSNMSN': 1.,
          'gainMSNGTA': 1.,
          'gainMSNGTI': 1.,
          'gainMSNGPi': 1.,
                  
          'gainFSIMSN': 1.,
          'gainFSIFSI': 1.,
                 
          'gainSTNMSN': 1.,
          'gainSTNFSI': 1.,
          'gainSTNGTA': 1.,
          'gainSTNGTI': 1.,
          'gainSTNGPi': 1.,
                 
          'gainGTAMSN': 1.,
          'gainGTAFSI': 1., # change nothing
          'gainGTAGTA': 1/5.,
          'gainGTAGTI': 1/5., #
                  
          'gainGTISTN': 1., # change nothing
          'gainGTIGTA': 4/5., #
          'gainGTIGTI': 4/5.,
          'gainGTIGPi': 1.,
###############################################################################         
          'inDegCSNMSN': 0.33333, # Max = 1.       | [1, 342]
          'inDegCSNFSI': 0.33333, # Max = 1.       | [1, 250]
                  
          'inDegPTNMSN': 0.33333, # Max = 1.       | [1, 5]
          'inDegPTNFSI': 0.33333, # Max = 1.       | [1, 5]
          'inDegPTNSTN': 0.33333, # Max = 0.386    | [1, 100]
                
          'inDegCMPfMSN':0.33333, # Max = 0.61     | [0.00325, 16]
          'inDegCMPfFSI':0.33333, # Max = 1.       | [0.1616, 170]
          'inDegCMPfSTN':20., # Max = 1.       | [1.11, 85]
          'inDegCMPfGTA':20., # Max = 1.       | [0.34, 27]
          'inDegCMPfGTI':20., # Max = 1.       | [0.34, 27]
          'inDegCMPfGPi':0.33333, # Max = 1.       | [0.6, 78.66]
                 
          'inDegMSNMSN': 0.33333, # Max = 1.       | [1, 209.78] 
          'inDegMSNGTA': 0.33333, # Max = 1.           | [105.37, 2644]  ****500
          'inDegMSNGTI': 0.33333, # Max = 1.           | [105.37, 2644]  ****500
          'inDegMSNGPi': 0.33333, # Max = 1.       | [151.66, 2644]
                  
          'inDegFSIMSN': 0.33333,      # /!\ Max = 0.011| [15, 53] too low = MSN distabilization  ***20
          'inDegFSIFSI': 0.33333, # Max = 1.       | [1, 33]
                 
          'inDegSTNMSN': 0.33333, # Max = 1.       | [0.00049, 0.01]
          'inDegSTNFSI': 0.33333, # Max = 0.517    | [0, 2.24]
          'inDegSTNGTA': 4., # /!\ Max = 0.023      | [4, 8]  ****4
          'inDegSTNGTI': 4., # /!\ Max = 0.023      | [4, 8]   ****4
          'inDegSTNGPi': 0.33333, # /!\ Max = 0.048| [1, 8]
                 
          'inDegGTAMSN': 0.33333, # Max = 0.428    | [0.001518, 0]
          'inDegGTAFSI': 0.33333, # /!\ Max = 0.176| [0.08, 5]
          'inDegGTAGTA': 2.,# /!\ Max = 0.079       | [1, 5]
          'inDegGTAGTI': 2.,# /!\ Max = 0.063      | [1, 5]
                  
          'inDegGTISTN': 4., # Max = 1.       | [3.25, 20]
          'inDegGTIGTA': 8., # /!\ Max = 0.079     | [0.84, 20] 
          'inDegGTIGTI': 8., # /!\ Max = 0.063     | [0.84, 20]
          'inDegGTIGPi': 0.33333, # Max = 1.      | [1.47, 20]
###############################################################################      
          'cTypeCSNMSN':    'focused', # defining connection types for channel-based models (focused or diffuse) based on LG14
          'cTypeCSNFSI':    'focused',
          
          'cTypePTNMSN':    'focused',
          'cTypePTNFSI':    'focused',
          'cTypePTNSTN':    'focused',
          
          'cTypeCMPfMSN':   'diffuse',
          'cTypeCMPfFSI':   'diffuse',
          'cTypeCMPfSTN':   'diffuse',
          'cTypeCMPfGTI':   'diffuse', #
          'cTypeCMPfGTA':   'diffuse', #
          'cTypeCMPfGPi':   'diffuse',
          
          
          'cTypeMSNMSN':    'diffuse',
          'cTypeMSNGTI':    'focused', #
          'cTypeMSNGTA':    'focused', #
          'cTypeMSNGPi':    'focused',
          
          'cTypeFSIMSN':    'diffuse',
          'cTypeFSIFSI':    'diffuse',
          
          'cTypeSTNMSN':    'diffuse',
          'cTypeSTNFSI':    'diffuse',
          'cTypeSTNGTI':    'diffuse', #
          'cTypeSTNGTA':    'diffuse', #
          'cTypeSTNGPi':    'diffuse',
          
          'cTypeGTAMSN':    'diffuse', #
          'cTypeGTAFSI':    'diffuse', #
          'cTypeGTAGTA':    'diffuse', #
          'cTypeGTAGTI':    'diffuse', #
          
          'cTypeGTISTN':    'focused', #
          'cTypeGTIGTA':    'diffuse', #
          'cTypeGTIGTI':    'diffuse', #
          'cTypeGTIGPi':    'diffuse',}
