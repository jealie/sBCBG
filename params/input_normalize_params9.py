# Maximal hypersphere of parameters giving a plausibility score of 14/14

the_scale = 4.

params = {
  # original LG14 parameterization
  'LG14modelID':9 ,
  # hypersphere parameterization
  'IeMSN': 24.5 ,
  'IeFSI': 8. ,
  'IeSTN': 9.5 ,
  'IeGPe': 12. ,
  'IeGPi': 11. ,
  # original number of neurons, possibly scaled
  'nbMSN':  the_scale * 2644. ,
  'nbFSI':  the_scale * 53. ,
  'nbSTN':  the_scale * 8. ,
  'nbGPe':  the_scale * 25. ,
  'nbGPi':  the_scale * 14. ,
  'nbCSN':  the_scale * 3000. ,
  'nbPTN':  the_scale * 3000. ,
  'nbCMPf': the_scale * 3000. ,
  # duration of a simulation (in ms)
  'simDuration': 10000.,
  ## parameters for the input screen: CSN
  #'varying': 'CSN',
  #'PActive_ini': 0.,
  #'PActive_max': 0.3,
  ## parameters for the input screen: PTN
  #'varying': 'PTN',
  #'PActive_ini': 0.,
  #'PActive_max': 0.3,
  ## parameters for the input screen: CM/Pf
  #'varying': 'CMPf',
  #'PActive_ini': 0.,
  #'PActive_max': 0.3,
  # parameters for the input screen: all
  'varying': 'all',
  'PActive_ini': 0.,
  'PActive_max': 0.3,
}
