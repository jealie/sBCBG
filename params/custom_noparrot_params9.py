# Hand-tuned parameters with custom inDegree values for original parameterization #9
# CM/Pf are NOT modeled with parrot neurons
# Degrees of freedom explored: gains for all nuclei + tonic input for GPe and GPi


params   =       {'LG14modelID':          9,
                  'nbMSN':            2644.,
                  'nbFSI':              53.,
                  'nbSTN':               8.,
                  'nbGPe':              25.,
                  'nbGPi':              14.,
                  'nbCSN':            3000.,
                  'nbPTN':             100.,
                  'nbCMPf':              9.,
                  'GMSN':              4.37,
                  'GFSI':               1.3,
                  'GSTN':              1.38,
                  'GGPe':               1.3,
                  'GGPi':                1.,
                  'IeGPe':              13.,
                  'IeGPi':              11.,
                  'RedundancyType':   'inDegreeAbs', # by default all axons are hypothesized to target each dendritic tree at 3 different locations
                  'redundancyCSNMSN':       100.,
                  'redundancyPTNMSN':         1.,
                  'redundancyCMPfMSN':        1.,
                  'redundancyFSIMSN':        30., # 30 : according to Humphries et al. 2010, 30-150 FSIs->MSN
                  'redundancyMSNMSN':        70., # 70 = 210/3 : according to Koos et al. 2004, cited by Humphries et al., 2010, on avg 3 synpase per MSN-MSN connection
                  'redundancySTNMSN':         0.,
                  'redundancyGPeMSN':         0.,
                  'redundancyCSNFSI':        50.,
                  'redundancyPTNFSI':         1.,
                  'redundancySTNFSI':         2.,
                  'redundancyGPeFSI':        25.,
                  'redundancyCMPfFSI':        9.,
                  'redundancyFSIFSI':        15., # 15 : according to Humphries et al., 2010, 13-63 FSIs->FSI
                  'redundancyPTNSTN':        25.,
                  'redundancyCMPfSTN':        9.,
                  'redundancyGPeSTN':        25.,
                  'redundancyCMPfGPe':        9.,
                  'redundancySTNGPe':         8.,
                  'redundancyMSNGPe':      2644.,
                  'redundancyGPeGPe':        25.,
                  'redundancyMSNGPi':      2644.,
                  'redundancySTNGPi':         8.,
                  'redundancyGPeGPi':        23.,
                  'redundancyCMPfGPi':        9.,
                  'cTypeCMPfMSN': 'focused', # LG14: diffuse
                  'cTypeMSNMSN':  'focused', # LG14: diffuse
                  'cTypeGPeFSI':  'focused', # LG14: diffuse
                  'cTypeCMPfFSI': 'focused', # LG14: diffuse
                  'cTypeCMPfSTN': 'focused', # LG14: diffuse
                  'cTypeGPeSTN':  'diffuse', # LG14: focused
                  'cTypeCMPfGPe': 'focused', # LG14: diffuse
                  'cTypeCMPfGPi': 'focused', # LG14: diffuse
                  'parrotCMPf' :      False,
                  }

