class Params(object):
  mk      =  0.497 
  mksig   =  0.000016
  mpi     =  0.134
  mpisig  =  0.00000018
  meta    =  0.547862
  metasig =  0.000017
  alpha   =  0.337848   # alpha strong at the charm mass
  beta    = -0.0293845  # alpha strong beta function
  gamma   = -0.332003   # anomalous dimension of quark mass
  gf      =  2.336925e-05
  mu      =  0.547862
  L85     = -0.46e-3
  L85sig  =  0.20e-3
  L64     =  0.28e-3
  L64sig  =  0.17e-3
  F0      =  0.0803
  F0sig   =  0.0006

import numpy as np
import sys
import os 
from icecream import ic

rng = np.random.default_rng(seed=None) 

class HipsofCobra():
  # Read in constituent form factors in order to derive 
  #   G form fractors. 
  with open('input/hips_gammapi.txt', 'r') as file:
    gammapi_sl = eval(file.read().replace('C', 'c') )
  with open('input/hips_deltapi.txt', 'r') as file:
    deltapi_sl = eval(file.read().replace('C', 'c') )
  with open('input/hips_thetapi.txt', 'r') as file:
    thetapi_sl = eval(file.read().replace('C', 'c') )
  with open('input/hips_gammaK.txt', 'r') as file:
    gammaK_sl = eval(file.read().replace('C', 'c') )
  with open('input/hips_deltaK.txt', 'r') as file:
    deltaK_sl = eval(file.read().replace('C', 'c') )
  with open('input/hips_thetaK.txt', 'r') as file:
    thetaK_sl = eval(file.read().replace('C', 'c') )

  # Read in C and D functions (canonical Omnes Solutions)
  #   in order to directly compute G form factors. 
  with open('input/hips_c1.txt', 'r') as file:
    c1_sl = eval(file.read().replace('C', 'c') )
  with open('input/hips_c2.txt', 'r') as file:
    c2_sl = eval(file.read().replace('C', 'c') )
  with open('input/hips_d1.txt', 'r') as file:
    d1_sl = eval(file.read().replace('C', 'c') )
  with open('input/hips_d2.txt', 'r') as file:
    d2_sl = eval(file.read().replace('C', 'c') )

  def __init__(self, clist, method):
    self.clist  = clist
    self.method = method 
    _l85   = rng.normal( Params.L85 , Params.L85sig )
    _l64   = rng.normal( Params.L64 , Params.L64sig )
    _mpi   = rng.normal( Params.mpi , Params.mpisig )
    _mk    = rng.normal( Params.mk  , Params.mksig )
    _meta  = rng.normal( Params.meta, Params.metasig )
    _F0    = rng.normal( Params.F0  , Params.F0sig )
    xi_hat = self.clist[0]
    xi_s   = self.clist[1]
    xi_g   = self.clist[2] * Params.alpha**2 / (3.0 * np.pi * Params.beta) 
  
  def Gpi0_direct(self):
    return 0

  

a = HipsofCobra([1,1,1], 'derived')  
print( a.clist ) 
a.Gpi0_direct()

