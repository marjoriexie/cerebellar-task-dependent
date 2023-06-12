# -*- coding: utf-8 -*-
"""
This file is part of the cerebellar-task-dependent repository (Xie et al. 2023)
Copyright (C) 2023 Marjorie Xie and Ashok Litwin-Kumar
See README for more information

A small library for drawing functions from Gaussian processes with squared exponential 
covariance function. 

"""
import numpy as np
import kernellib
import sphlib

def draw_gp_functions_cholesky(length_scale, D, P=5000, N_out=1, 
                                   noise_std=0., 
                                   normalize_target=True):
    '''
    Draw functions from a Gaussian process (GP) 
    using Cholesky decomposition. 
    See C. E. Rasmussen & C. K. I. Williams, 
    Gaussian Processes for Machine Learning, the MIT Press, 2006. https://gaussianprocess.org/gpml/chapters/RW.pdf

    Parameters
    ----------
    length_scale : length scale of the GP covariance function 
    D : dimension of the space of task variables 
    P : int, number of data points. The default is 5000.
    N_out : int, number of GP functions to draw. The default is 1.
    noise_std : standard deviation of noise in the output. The default is 0..
    normalize_target : bool, True - divide the output by its standard deviation. The default is True.

    Returns
    -------
    X : input data (D, P)
    y: outputs (N_out, P)

    '''
    
    # Draw random data points on a (D-1) sphere
    X = np.random.randn(D, P)
    if D > 1:
        X = X / np.sqrt(np.sum(X ** 2, 0, keepdims=True))

    # generate covariance matrix 
    C = kernellib.sqr_exp_kernel(X.T @ X, length_scale)
    # perform the decomposition 
    constant = 1e-10 # you can add noise by changing this scalar diagonal term 
    L = np.linalg.cholesky(C + constant * np.eye(P)) 
    y = np.dot(L, np.random.randn(P, N_out)).T 
    
    if normalize_target:
        # make the standard deviation of the output 1 but allow arbitrary mean
        y = y / y.std()
        
    
    return X, y
    
    
def draw_gp_functions_coeffvar(coeffvars, length_scale, D, P, Nk, N_out=1,
                                   noise_std=0., 
                                   normalize_target=True):
    '''
    Draw functions from a GP using the variance of the coefficients 
    of the  GP's covariance function (squared exponential function) 
    in the spherical harmonic basis.
    See Appendix section C.4 for the corresponding equations. 
    
    Parameters
    ----------
    coeffvars : variance of the coefficients of the square exponential covariance function
    length_scale : length scale of the GP covariance function 
    D : dimension of the space of task variables 
    P : int, number of data points 
    N_out : int, number of GP functions to draw. The default is 1.
    Nk : int, number of frequency modes (indexed by k) to represent functions 
        in the spherical harmonic basis. The default is 10.
    noise_std : standard deviation of noise in the output. The default is 0..
    normalize_target : bool, True - divide the output by its standard deviation. The default is True.

    Returns
    -------
    X : input data (D, P)
    y: outputs (N_out, P)
    coeff : coefficients of a particular function drawn from the GP

    '''
        
    # Draw random data points on a (D-1) sphere
    X = np.random.randn(D, P)
    if D > 1:
        X = X / np.sqrt(np.sum(X ** 2, 0, keepdims=True))

    #Convert data from Cartesian to angular coordinates
    theta = np.zeros([D - 1, P])
    for qi in range(P):
        theta[:, qi] = sphlib.to_ang(X[:, qi])  

    H, k_arr = sphlib.create_sphfeatures_matrix(theta, Nk)

    spectrum = coeffvars[k_arr]
    spectrum[spectrum<0] = 0 #if there are numerical precision issues
    coeff = np.sqrt(spectrum[:, np.newaxis]) * np.random.randn(len(spectrum), N_out)
    y = coeff.T @ H
    
    if normalize_target:
        # make the standard deviation of the output 1 but allow arbitrary mean
        y = y / y.std()
        
    return X, y, coeff


    
        
    
