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
  def __init__(self, clist, Pname, method):
    _npi = 1
    _nK  = np.sqrt(3.0)/2.0

    assert method=='derived' or method=='direct' , \
      " Method must be either 'derived' or 'direct'. "
    assert Pname=='pi' or Pname=='K' , \
      " Pname must be either 'pi' or 'K'. "

    self.clist  = clist
    self.method = method 
    self.xi_hat = self.clist[0]
    self.xi_s   = self.clist[1]
    self.xi_g   = self.clist[2]*Params.alpha**2/(3.0*np.pi*Params.beta) 
    Gpi_deriv_mean =  self.xi_g 
    Gpi_deriv_std  =  0.0191309 # Unc. from (mpi/4*pi*F_pi)^2
    GK_deriv_mean  = -0.536731
    GK_deriv_std   =  0.239351 # Unc. from (mK/4*pi*F_pi)^2

    # Extract mean and stdev of Gpi0 and GK0 values. 
    _gsamples = 1000
    _gpilist  = [self.Gpi0() for i in range(_gsamples)]
    _gKlist   = [self.GK0()  for i in range(_gsamples)]
    gpi0mean = np.average(_gpilist)
    gpi0std  = np.std(_gpilist)
    gK0mean  = np.average(_gKlist)
    gK0std   = np.std(_gKlist)
    
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

    slist = c1_sl[0]
    number_of_inds  = len(c1_sl[0]) 
    number_of_iters = len(c1_sl[1])
    # Compute C and D derivatives
    c1_deriv = [ 0.5*( 
      (c1_sl[1][iter][1]-c1_sl[1][iter][0]) / (slist[1]-slist[0]) + \
      (c1_sl[1][iter][2]-c1_sl[1][iter][1]) / (slist[2]-slist[1])
    ) for iter in range(number_of_iters) ]
    c2_deriv = [ 0.5*( 
      (c2_sl[1][iter][1]-c2_sl[1][iter][0]) / (slist[1]-slist[0]) + \
      (c2_sl[1][iter][2]-c2_sl[1][iter][1]) / (slist[2]-slist[1])
    ) for iter in range(number_of_iters) ] 
    d1_deriv = [ 0.5*( 
      (d1_sl[1][iter][1]-d1_sl[1][iter][0]) / (slist[1]-slist[0]) + \
      (d1_sl[1][iter][2]-d1_sl[1][iter][1]) / (slist[2]-slist[1])
    ) for iter in range(number_of_iters) ] 
    d2_deriv = [ 0.5*( 
      (d2_sl[1][iter][1]-d2_sl[1][iter][0]) / (slist[1]-slist[0]) + \
      (d2_sl[1][iter][2]-d2_sl[1][iter][1]) / (slist[2]-slist[1])
    ) for iter in range(number_of_iters) ]
    
    self.G_sl = [slist, []]   
    for iter in range(number_of_iters):
      thetapi0_dummy = thetaPi0()
      thetaK0_dummy = thetaK0()
      Gpi0_dummy = self.Gpi0()
      GK0_dummy  = self.GK0()
      Gpi_deriv_dummy = rng.normal( Gpi_deriv_mean, Gpi_deriv_std )
      GK_deriv_dummy  = rng.normal( GK_deriv_mean, GK_deriv_std )

      Qpi0 = _npi*Gpi0_dummy
      QK0  = _nK*GK0_dummy
      if self.method=='derived':
        Qpi1 = _npi*Gpi_deriv_dummy-\
                  self.xi_g*(
                    _npi*thetapi0_dummy*c1_deriv[iter]+\
                    _nK*thetaK0_dummy*d1_deriv[iter]          
                  )
        QK1  = _nK*GK_deriv_dummy-\
                  self.xi_g*(
                    _nK*thetaK0_dummy*d2_deriv[iter]+\
                    _npi*thetapi0_dummy*c2_deriv[iter]
                  )
      else: 
        Qpi1 = _npi*Gpi_deriv_dummy-\
                c1_deriv[iter]*Gpi0_dummy-\
                d1_deriv[iter]*GK0_dummy
        QK1  = _nK*GK_deriv_dummy-\
                c2_deriv[iter]*Gpi0_dummy-\
                d2_deriv[iter]*GK0_dummy
      
      if Pname=='pi':
        gvals = [
          c1_sl[1][iter][i]*(Qpi0+slist[i]*Qpi1) +\
          d1_sl[1][iter][i]*(QK0+slist[i]*QK1)/_npi
          for i in range(number_of_inds)]
      else: 
        gvals = [
          c2_sl[1][iter][i]*(Qpi0+slist[i]*Qpi1) +\
          d2_sl[1][iter][i]*(QK0+slist[i]*QK1)/_nK
          for i in range(number_of_inds)]
      self.G_sl[1].append(gvals)


    

  def Gpi0(self, MeanQ=False):
    return self.xi_g*thetaPi0(MeanQ=MeanQ)-\
      (self.xi_hat+(1.0-Params.gamma)*self.xi_g)*GammaPi0(MeanQ=MeanQ)-\
      (self.xi_s+(1.0-Params.gamma)*self.xi_g)*DeltaPi0(MeanQ=MeanQ)

  def GK0(self, MeanQ=False):
    return self.xi_g*thetaK0(MeanQ=MeanQ)-\
      (self.xi_hat+(1.0-Params.gamma)*self.xi_g)*GammaK0(MeanQ=MeanQ)-\
      (self.xi_s+(1.0-Params.gamma)*self.xi_g)*DeltaK0(MeanQ=MeanQ)


a = HipsofCobra([1,1,1], 'pi', 'derived')  
ic( a.G_sl[1][1][:10] ) 