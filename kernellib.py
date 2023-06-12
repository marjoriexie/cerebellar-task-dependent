# -*- coding: utf-8 -*-
"""
This file is part of the cerebellar-task-dependent repository (Xie et al. 2023)
Copyright (C) 2023 Marjorie Xie and Ashok Litwin-Kumar
See README for more information

Library which includes functions for computing the kernel of a ReLU network 
analytically and also the generalization error.. 

"""
import numpy as np
import scipy.special as special
from scipy.stats import norm
from scipy.optimize import fsolve
import scipy.integrate as integrate


def sqr_exp_kernel(t, length_scale):
    '''
    Squared exponential kernel function. This is the covariance function used 
    in our GP tasks.
    
    Parameters
    ----------
    t : Overlap between two input vectors 
    length_scale : length scale of exponential decay function

    '''
    return np.exp(-(1 - t) / length_scale ** 2)


def g1(s,T):
    '''
    Helper function for integrand_relu().

    Parameters
    ----------
    s : scalar
    T : scalar 
  
    '''
    result = s * (1 / np.sqrt(2 * np.pi)) * np.exp(-T ** 2 / 2)
    return result

def g2(s,T):
    '''
    Helper function for integrand_relu().

    Parameters
    ----------
    s : scalar
    T : scalar 
  
    '''
    return s * (1 / 2) * special.erfc(T / (np.sqrt(2)))

# same as integrand() except accomodates negative rho values
def integrand_relu(z, args):
    '''
    Helper function for relu_kernel_intquad().
     Parameters
     ----------
     z : variable being integrated over
     args : list containing rho () and thresh (threshold of activation)
    Returns the integrand for computing analytically the kernel for a network with a ReLU 
    units in the expansion layer. 
      
    '''
    Pz = norm.pdf(z, scale=1)
    thresh = args[1] # ReLU threshold
    rho = args[0]

    if rho < 0:
        rho_negative = 1 # flag
        rho = np.abs(rho) # this is so we can take sqrt(rho) in the expressions below
    else:
        rho_negative = 0

    a = np.sqrt(1 - rho)
    b = np.sqrt(rho) * z - thresh
    c = -np.sqrt(rho) * z - thresh
    T1 = (thresh - np.sqrt(rho) * z) / np.sqrt(1 - rho)
    T2 = (thresh + np.sqrt(rho) * z) / np.sqrt(1 - rho)

    if rho_negative:
        y_integrals = (g1(a,T1) + g2(b,T1)) * (g1(a,T2) + g2(c,T2))
    else: # positive rho values
        y_integrals = (g1(a,T1) + g2(b,T1)) ** 2

    return Pz * y_integrals


def relu_kernel_intquad(rho, thresh, meansub=1):
    '''
    Kernel function for an infinite-width network with ReLU expansion layer
    For the equations, see Appendix section B) Dot-product kernels with arbitrary threshold

    Parameters
    ----------
    rho : overlap in the input layer 
    thresh : threshold of the ReLU activation function
    meansub : bool, whether to mean-subtract the kernel 

    Returns
    -------
    overlap : overlap in the expansion layer

    '''
    args = [rho, thresh]
    lo, hi = -5, 5 # bounds of integration for z
    E_mdotm, err = integrate.quad(integrand_relu, lo, hi, args=args)
    
    if meansub:
        Em = 1 / np.sqrt(2 * np.pi) * np.exp(-thresh ** 2 / 2) -\
          (thresh / 2) * special.erfc(thresh / np.sqrt(2))
          #normalization term subtracts E[m1] * E[m2]
        overlap = E_mdotm - Em ** 2 # mean-subtracted overlap 
    else:
        overlap = E_mdotm
    
    return overlap


def gen_err_Bordelon(coeff, eig, P, lam, noise_std):
    '''
    Theoretical expression for the expected squared error, following
    Bordelon, Canatar, Pehlevan papers. Specifically, we follow
    "Spectral Bias and Task-Model Alignment Explain Generalization in
    Kernel Regression and Infinitely Wide Neural Networks" by Canatar,
    Bordelon, Pehlevan. Nature Communications, 2021.

    Note: Nalpha refers to the number of basis functions
    Note: coefficients have different normalization between 
      Bordelon / Canatar papers

    Parameters
    ==========
    coeff : np.ndarray (Nalpha,), coefficients of target
    eig : np.ndarray (Nalpha,), eigenvalues
    lam : float, regularization parameter
    P : float, number of samples

    Returns
    =======
    err : float, expected generalization error
    b : array of terms in the sum over k in the main Eg expression
    '''
    assert len(coeff) == len(eig), "coeff and eig should be same shape"
    assert lam >= 0, "lam must be >= 0"
    assert P > 0, "P must be > 0"

    def kappa_eqn(kappa, eig, lam, P):
        '''
        root of this equation determines kappa
        '''
        return lam + kappa * (np.sum(eig / (eig * P + kappa)) - 1)

    def kappa_eqn_prime(kappa, eig, lam, P):
        '''
        derivative of kappa_eqn
        '''
        return -1 + np.sum(eig / (eig * P + kappa)) - \
          kappa * np.sum(eig / (eig * P + kappa) ** 2)

    def chi_eqn(kappa, eig, lam, P):
        # formula for chi
        a = eig ** 2 / (eig * P + kappa) ** 2
        chi = P * np.sum(a)
        return chi, a

    # solve kappa_eqn = 0 with initial guess 1
    kappa = fsolve(kappa_eqn, 1, (eig, lam, P), fprime=kappa_eqn_prime)[0]
    print(f'kappa = {kappa}')
    chi, a = chi_eqn(kappa, eig, lam, P)
    print(f'chi = {chi}')
    
    b = (coeff ** 2 * kappa ** 2 + noise_std ** 2 * P * eig ** 2) \
                      / (eig * P + kappa) ** 2 
                      
    ##weighting of frequency components that depends on P but not the task 
    beta = (1 / (1 - chi)) * ( (kappa ** 2) / (eig * P + kappa) ** 2 )
    
    Eg = 1 / (1 - chi) * np.sum(b) #theoretical error 
    
    return Eg + noise_std ** 2, b, a, beta, kappa



