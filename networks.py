# -*- coding: utf-8 -*-
"""
This file is part of the cerebellar-task-dependent repository (Xie et al. 2023)
Copyright (C) 2023 Marjorie Xie and Ashok Litwin-Kumar
See README for more information

Library of functions for generating activity in the granule cell layer of the model and for 
learning the readout weights. 

 
"""

import numpy as np
import scipy.special as special
from sklearn.linear_model import Ridge

def get_regression_coefficients(H, y, lam):
    '''
    Implements ridge regression. To implement least squares, set lam = 0.

    Parameters
    ----------
    H : (M, P) array, the design matrix
    y : (N_out, P) array of targets (regressand), where N_out is the number of outputs and P is the number of patterns
    lam : regularization parameter 

    Returns
    -------
    W : weights to be learned through ridge regression
    lam : regularization parameter 

    '''
    clf = Ridge(alpha=lam, fit_intercept=False)
    clf.fit(H.T, y.T)
    W = clf.coef_.T
    return W, lam

def relu(H, f):
    '''
    Apply relu nonlinearity to granule cell currents given a desired coding level f. 
    Assumes activation function is ReLU. 

    Parameters
    ----------
    H : (M, P) array of granule cell layer currents
    f : float, desired coding level of the granule cell layer 
    M : int, number of units in the granule cell layer 
    P : int, number of patterns to generate activations for

    Returns
    -------
    H_nonlin : (M, P) array of granule cell activations after relu is applied
    thresh: float, threshold that generates the desired coding level in H

    '''
    M, P = H.shape
    thresh = np.sort(H.flatten())[int((1-f)*(M*P))]
    H = H - thresh
    H_nonlin = H * (H > 0)
    
    return H_nonlin, thresh


def f_to_thresh(f, var=1): 
    '''
    # The coding level, f, is the probability that a given unit in the granule cell
    # layer is greater than zero. 
    # For a ReLU unit in the granule cell layer that is a Gaussian random variable, 
    # this function determines the threshold that makes the unit active with probability f.
    # It is assumed that the units have variance 1.  

    Parameters
    ----------
    f : desired coding level, or probability of a granule cell being active 
    var : variance of a unit in the granule cell layer. This function assumes that the variance is 1. 

    '''
    return np.sqrt(2 * var) * special.erfcinv(2 * f)