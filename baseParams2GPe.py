# Simulation defaults
# these defaults can be overrided via a custom python parameter file or via the commandline (in this order: commandline arguments take precedence over customParams.py, which take precendence over the defaults defined here)
params   =       {'durationH':           '04', # used by Sango
                  'durationMin':         '00', # used by Sango
                  'nbnodes':              '1', # used by K
                  'nestSeed':              20, # nest seed (affects input poisson spike trains)
                  'pythonSeed':            10, # python seed (affects connection map)
                  'nbcpu':                  1,
                  'whichTest':   'testFullBG',
                  'nbCh':                   1,
                  'LG14modelID':           10,
                  'tSimu':              5000.,
                  
                  'nbMSN':              2644.,
                  'nbFSI':                53.,
                  'nbSTN':                 8.,
                  'nbGTI':                20., # 84% GPe --> 80% GPe ( SATO 2000 )
                  'nbGTA':                 5., # 16% GPe --> 20% GPe ( SATO 2000 )
                  'nbGPi':                14.,
                  'nbCSN':              3000.,
                  'nbPTN':               100.,
                  'nbCMPf':             3000.,
                  
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
                  'gainGTAFSI': 1.,
                  'gainGTAGTA': 1.,
                  'gainGTAGTI': 1.,
                      
                  'gainGTISTN': 1.,
                  'gainGTIGTA': 1.,
                  'gainGTIGTI': 1.,
                  'gainGTIGPi': 1.,
                  
                  'IeMSN':                 0.,
                  'IeFSI':                 0.,
                  'IeSTN':                 0.,
                  'IeGTI':                 0.,
                  'IeGTA':                 0.,
                  'IeGPi':                 0.,
                  
                  'inDegCSNMSN':      0.33333, # inDegrees are all set to 1/3 of maximal input population
                  'inDegPTNMSN':      0.33333,
                  'inDegCMPfMSN':     0.33333,
                  'inDegFSIMSN':      0.33333,
                  'inDegMSNMSN':      0.33333,
                  'inDegSTNMSN':      0.33333,
                  'inDegGTAMSN':      0.33333, #
                  'inDegCSNFSI':      0.33333,
                  'inDegPTNFSI':      0.33333,
                  'inDegSTNFSI':      0.33333,
                  'inDegGTAFSI':      0.33333, #
                  'inDegCMPfFSI':     0.33333, 
                  'inDegFSIFSI':      0.33333, 
                  'inDegPTNSTN':      0.33333, 
                  'inDegCMPfSTN':     0.33333, 
                  'inDegGTISTN':      0.33333, #
                  'inDegCMPfGTI':     0.33333, #
                  'inDegCMPfGTA':     0.33333, #
                  'inDegSTNGTI':      0.33333, #
                  'inDegSTNGTA':      0.33333, #
                  'inDegMSNGTI':      0.33333, #
                  'inDegMSNGTA':      0.33333, #
                  'inDegGTIGTA':      (4/5.)*0.33333, #
                  'inDegGTIGTI':      (4/5.)*0.33333, #
                  'inDegGTAGTA':      (1/5.)*0.33333, #
                  'inDegGTAGTI':      (1/5.)*0.33333, #
                  'inDegMSNGPi':      0.33333,
                  'inDegSTNGPi':      0.33333,
                  'inDegGTIGPi':      0.33333, #
                  'inDegCMPfGPi':     0.33333,
                  
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
                  'cTypeGTIGPi':    'diffuse',
                  'parrotCMPf' :      True,}

