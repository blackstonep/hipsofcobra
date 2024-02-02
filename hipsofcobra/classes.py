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

# Functions for computing s=0 values for all form factors. 
#
# MeanQ = True if we want to central prediction of FF, as opposed to a sampling from distributions. 
def GammaPi0(MeanQ=False): 
  _mu = Params.mu
  if MeanQ:
    _l85   = Params.L85
    _l64   = Params.L64
    _mpi   = Params.mpi
    _mk    = Params.mk
    _meta  = Params.meta
    _F0    = Params.F0
  else: 
    _l85   = rng.normal( Params.L85 , Params.L85sig  )
    _l64   = rng.normal( Params.L64 , Params.L64sig  )
    _mpi   = rng.normal( Params.mpi , Params.mpisig  )
    _mk    = rng.normal( Params.mk  , Params.mksig   )
    _meta  = rng.normal( Params.meta, Params.metasig )
    _F0    = rng.normal( Params.F0  , Params.F0sig   )
  
  return _mpi**2 + ( _mpi**4 / _F0**4 ) * ( 
    (1.0/32.0/np.pi**2) * (
      (8.0/9.0)+np.log(_mpi**2/_mu**2)- \
        (1.0/9.0)*np.log(_meta**2/_mu**2)
    ) + 8*_l85 + 16*_l64 
  )

def DeltaPi0(MeanQ=False): 
  _mu = Params.mu
  if MeanQ:
    _l85   = Params.L85
    _l64   = Params.L64
    _mpi   = Params.mpi
    _mk    = Params.mk
    _meta  = Params.meta
    _F0    = Params.F0
  else: 
    _l85   = rng.normal( Params.L85 , Params.L85sig  )
    _l64   = rng.normal( Params.L64 , Params.L64sig  )
    _mpi   = rng.normal( Params.mpi , Params.mpisig  )
    _mk    = rng.normal( Params.mk  , Params.mksig   )
    _meta  = rng.normal( Params.meta, Params.metasig )
    _F0    = rng.normal( Params.F0  , Params.F0sig   )
  
  return (_mpi**2*(_mk**2-0.5*_mpi**2))/_F0**2 * (
    -1.0/(72.0*np.pi**2)*(1.0+np.log(_meta**2/_mu**2))+16*_l64
  )

def GammaK0(MeanQ=False): 
  _mu = Params.mu
  if MeanQ:
    _l85   = Params.L85
    _l64   = Params.L64
    _mpi   = Params.mpi
    _mk    = Params.mk
    _meta  = Params.meta
    _F0    = Params.F0
  else: 
    _l85   = rng.normal( Params.L85 , Params.L85sig  )
    _l64   = rng.normal( Params.L64 , Params.L64sig  )
    _mpi   = rng.normal( Params.mpi , Params.mpisig  )
    _mk    = rng.normal( Params.mk  , Params.mksig   )
    _meta  = rng.normal( Params.meta, Params.metasig )
    _F0    = rng.normal( Params.F0  , Params.F0sig   )
  
  return 0.5*_mpi**2+0.5*(_mpi**2/_F0**2)*(
    -_mpi**2/(32.0*np.pi**2)*np.log(_mpi**2/_mu**2)+\
    _meta**2/(32.0*np.pi**2)*np.log(_meta**2/_mu**2)+\
    (_mk**2-_mpi**2)*_l85
  ) + _mpi**2 *(_mk**2-0.5*_mpi**2) / (2.0*_F0**2) * (
    1.0/(72.0*np.pi**2)*(1.0+np.log(_meta**2/_mu**2))+\
    8*_l85+16*_l64
  )

def DeltaK0(MeanQ=False): 
  _mu = Params.mu
  if MeanQ:
    _l85   = Params.L85
    _l64   = Params.L64
    _mpi   = Params.mpi
    _mk    = Params.mk
    _meta  = Params.meta
    _F0    = Params.F0
  else: 
    _l85   = rng.normal( Params.L85 , Params.L85sig  )
    _l64   = rng.normal( Params.L64 , Params.L64sig  )
    _mpi   = rng.normal( Params.mpi , Params.mpisig  )
    _mk    = rng.normal( Params.mk  , Params.mksig   )
    _meta  = rng.normal( Params.meta, Params.metasig )
    _F0    = rng.normal( Params.F0  , Params.F0sig   )
  
  return (_mk**2-0.5*_mpi**2)+0.5*_mpi**2/_F0**2*(
    _mpi**2/(32.0*np.pi**2)*np.log(_mpi**2/_mu**2)-\
    _meta**2/(32.0*np.pi**2)*np.log(_meta**2/_mu**2)-\
    8*(_mk**2-_mpi**2)*_l85
  )+_mk**2*_mpi**2/_F0**2*(
    1.0/(36.0*np.pi**2)*(1.0+np.log(_meta**2/_mu**2))+\
    8*_l85+16*_l64
  )

def thetaPi0(MeanQ=False):
  return 2*GammaPi0(MeanQ=MeanQ)

def thetaK0(MeanQ=False):
  return 2*(DeltaK0(MeanQ=MeanQ)+GammaK0(MeanQ=MeanQ))

class HipsofCobra():
  def __init__(self, clist, method):
    assert method=='derived' or method=='direct' , \
      " Method must be either 'derived' or 'direct'. "

    self.clist  = clist
    self.method = method 
    _l85   = rng.normal( Params.L85 , Params.L85sig  )
    _l64   = rng.normal( Params.L64 , Params.L64sig  )
    _mpi   = rng.normal( Params.mpi , Params.mpisig  )
    _mk    = rng.normal( Params.mk  , Params.mksig   )
    _meta  = rng.normal( Params.meta, Params.metasig )
    _F0    = rng.normal( Params.F0  , Params.F0sig   )
    xi_hat = self.clist[0]
    xi_s   = self.clist[1]
    xi_g   = self.clist[2] * Params.alpha**2 / (3.0 * np.pi * Params.beta) 
    
    if self.method=='derived':
      # Read in constituent form factors in order to derive 
      #   G form fractors. 
      with open('input/hips_gammapi.txt', 'r') as file:
        self.gammapi_sl = eval(file.read().replace('C', 'c') )
      with open('input/hips_deltapi.txt', 'r') as file:
        self.deltapi_sl = eval(file.read().replace('C', 'c') )
      with open('input/hips_thetapi.txt', 'r') as file:
        self.thetapi_sl = eval(file.read().replace('C', 'c') )
      with open('input/hips_gammaK.txt', 'r') as file:
        self.gammaK_sl = eval(file.read().replace('C', 'c') )
      with open('input/hips_deltaK.txt', 'r') as file:
        self.deltaK_sl = eval(file.read().replace('C', 'c') )
      with open('input/hips_thetaK.txt', 'r') as file:
        self.thetaK_sl = eval(file.read().replace('C', 'c') )

    else:
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

  

a = HipsofCobra([1,1,1], 'derived')  
ic( a.clist ) 
ic( GammaPi0() )
ic( DeltaPi0() )
ic( GammaK0() )
ic( DeltaK0() )
ic( thetaPi0() )
ic( thetaK0() )
