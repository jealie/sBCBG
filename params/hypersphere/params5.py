# Maximal hypersphere of parameters giving a plausibility score of 14/14

the_scale = 4.

params = {
  # original LG14 parameterization
  'LG14modelID':5 ,
  # hypersphere parameterization
  'IeMSN': 23.75,
  'IeFSI': 4.,
  'IeSTN': 9.,
  'IeGPe': 10.,
  'IeGPi': 10.,
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
  'simDuration': 1000.
}