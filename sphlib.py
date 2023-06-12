# -*- coding: utf-8 -*-
"""
This file is part of the cerebellar-task-dependent repository (Xie et al. 2023)
Copyright (C) 2023 Marjorie Xie and Ashok Litwin-Kumar
See README for more information

Library of functions for working in the spherical harmonic basis.

The key functions to call are 
1. eigenvalue_spectrum() which computes the eigenvalues for any dot product kernel
2. create_sphfeatures_matrix(), which can be used to estimate the 
    coefficients of any function whose inputs are distributed uniformly on the unit sphere
    (see Eq 34 in Appendix), 
    by creating a design matrix of spherical harmonics and fitting the coefficients 
    to the outputs of the function. 
        

Depdendency structure:
    
- eigenvalue_spectrum() — see Eq. 33 and 34 in Appendix C.1
    - funk_kecke_integrand() — Eq. 39 in Appendix C.3
        - P(t, k, D) — Eq. 36 in Appendix C.2
        - weight_func()
        - kernel_func
    - sfc_area() — see Eq. 28 in Appendix C.1
- create_sphfeatures_matrix()
    - num_harmonics() — see Eq. 27 in Appendix C1
    - Y_ang()
-gen_single_freq_targets()
- Y_ang()
    - gegenbauer()
    - normalizing_factor()
        - lambda_j
    - abs_alpha_j()
    - g_alpha()
- Y_cart()
    - sin_theta_k()
    - cost_theta_k()
    - alpha_sum()
    - gegenbauer()
    - g_alpha()
    - lambda_j
- to_cart()
- to_ang()
- num_alpha()
    - num_harmonics()

"""

import numpy as np
import scipy.special as sp
import itertools
import sympy as sym
import scipy.integrate as integrate


def sfc_area(D):
    '''
    Surface area of S^{D-1}
    
    Dunkl & Xu, 4.1.3
    '''
    return 2 * np.pi ** (D / 2) / sp.gamma(D / 2)

def P(t, k, D):
    '''
    Evaluate the zonal harmonic polynomial of frequency k
    P_k(t) defined in Basri et al., 2020
    Helper function for funk_hecke_integrand()

    Parameters
    ----------
    t : overlap in the input space, can be scalar or vector
    k : harmonic mode
    D : dimension of input space (D >= 2)

    '''
    x = sym.Symbol('x')
    expr = sym.diff((1 - x**2)**(k + (D - 3) / 2), x, k)
    # take the kth derivative with respect to x
    f = sym.lambdify(x, expr, 'numpy')
    result = f(t) # evaluate the expression at x = t
    out = (-1)**k / 2**k * sp.gamma((D - 1) / 2) / sp.gamma(k + (D - 1) / 2) \
        * 1 / ((1 - t**2)**((D - 3) / 2)) * result
    return out

def weight_func(t, D):
    '''
    Weight function for gegenbauer polynomial defined in Basri et al., 2020.
    Helper function for funk_hecke_integrand()
    
    Parameters
    ----------
    t : overlap in input space
    D : dimension of input space
    '''
    return (1 - t ** 2) ** ((D - 3) / 2)

def funk_hecke_integrand(t, k, D, kernel_func, kernel_args):
    '''
    Computes Funk-Hecke integral for the eigenvalues of the kernel kernel_func.
    
    Parameters
    ----------
    t : overlap in input space
    k : frequency mode of spherical harmonic
    o    D : dimension of input space
    kernel_func : kernel function 
    kernel_args : parameters of the kernel function 

    '''
    return kernel_func(t, *kernel_args) * P(t, k, D) * weight_func(t, D)

def eigenvalue_spectrum(kernel_func, kernel_args, D, Nk, trange=[-1, 1]):
    '''
    Compute the first Nk eigenvalues of the kernel function in the spherical 
    harmonic basis. 
    Note: the eigenvalues are the same across rotations m, so only the 
    zonal harmonic eigenfunctions where m=0 are needed. 

    Parameters
    ----------
    kernel_func : kernel function whose eigenvalues we want to compute
    kernel_args : parameteres of kernel function
    D : input dimension
    Nk : number of frequency modes to compute eigenvalues for
    trange : overlap in input space for integration, default is [-1, 1].

    Returns
    -------
    eigs : first Nk eigenvalues of kernel function

    '''
    eigs = np.zeros(Nk)
    print('Computing kernel eigenvalue for each frequency k up to Nk = ' + str(Nk) + ' \n')
    for k in range(Nk):
        print('k = ', k)
        integral, err = integrate.quad(funk_hecke_integrand, trange[0], \
                                       trange[1], args=(k, D, kernel_func, 
                                                        kernel_args))
        eigs[k] = sfc_area(D - 1) * integral
    return eigs

def num_harmonics(D, k):
    '''
    Number of k-th degree linearly independent homogeneous harmonic polynomials 
    over input space of dimension D, from Efthimiou & Frye 2014 
    '''
    if k == 0:
        return 1
    else:
        return int(round((2 * k + D - 2) / k * sp.comb(k + D - 3, k - 1)))
    
def num_alpha(D, Nk):
    '''
    Total number of linearly independent homogeneous harmonic polynomials 
    including those within same degree k up to Nk-th degree
    '''
    N_alpha = 0
    for k in range(Nk):
        N_alpha += num_harmonics(D,k)
        
    return N_alpha
        

def to_ang(x):
    ''' 
    Convert cartesian coordinates to angular
    '''
    D = len(x)
    theta = np.zeros(D-1)

    rsq = x[0] ** 2
    for di in range(D-1):
        rsq = rsq + x[di + 1] ** 2
        if di == 0:
            theta[0] = np.mod(np.arctan2(x[0], x[1]), np.pi * 2)
        else:
            theta[di] = np.arccos(x[di + 1] / np.sqrt(rsq))

    return theta

def to_cart(theta):
    '''
    Convert angular coordinates to cartesian
    '''
    D = len(theta) + 1
    x = np.zeros(D)
    sintheta = np.sin(theta)
    for di in range(D - 1, -1, -1):
        if di == (D - 1):
            x[di] = np.cos(theta[di - 1])
        elif di == 0:
            x[di] = np.prod(sintheta)
        else:
            sinprod = np.prod(sintheta[di:(D - 1)])
            x[di] = sinprod * np.cos(theta[di - 1])

    return x

def cos_theta_k(x, k):
    '''
    Helper function for Y_cart()

    '''
    return x[k] / np.linalg.norm(x[:(k+1)])

def sin_theta_k(x, k):
    '''
    Helper function for Y_cart()
    '''
    return np.sqrt(1 - cos_theta_k(x, k) ** 2)

def gegenbauer(x, lam, n):
    '''
    Gegenbauer polynomial C_n^{\lambda} (x)
    '''
    return sp.eval_gegenbauer(n, lam, x)

def alpha_sum(i, arr):
    '''
    From Dai & Xu, used in Y_cart
    formula not verified
    '''
    return np.sum(arr[i:-1])

def abs_alpha_j(j, alpha):
    '''
    Dunkl & Xu, Theorem 4.1.4, called beta_j
    '''
    assert j >= 1 and j <= (len(alpha) - 1), "invalid range for j"
    # |alpha^j| = alpha_j + ... + alpha_d
    # shift by -1 for 0-based indexing, sum to 1 before last
    return np.sum(alpha[(j - 1):])

def lambda_j(j, alpha):
    '''
    Dunkl & Xu, in text before 4.1.1
    '''
    D = len(alpha)
    assert j >= 1 and j <= (D - 1), "invalid range for j"
    # lambda_j = |alpha^{j+1}| + (d - j - 1)/2
    return abs_alpha_j(j + 1, alpha) + (D - j - 1) / 2.

def g_alpha(alpha, t):
    '''
    Dunkl & Xu, Theorem 4.1.4
    alpha_d = 0 or 1
    '''
    if alpha[-1] == 0:
        # cos(alpha_{d-1} t)
        return np.cos(alpha[-2] * t)
    elif alpha[-1] == 1:
        # sin((alpha_{d-1} + 1) t)
        return np.sin((alpha[-2] + 1) * t)
    else:
        raise Exception("alpha[-1] unexpected")

def normalizing_factor(alpha):
    '''
    Normalizing constant for the spherical harmonics
    Dunkl & Xu, Theorem 4.1.4
    Does not take into account surface area of S^{D-1}
    '''
    D = len(alpha)
    if alpha[-2] + alpha[-1] > 0:
        b = 2
    else:
        b = 1
    ca = np.zeros(D - 2)
    for di in range(D - 2):
        j = di + 1
        alpha_j = alpha[di]
        lam = lambda_j(j, alpha)
        num = np.math.factorial(alpha_j) \
          * sp.poch((D - j + 1) / 2., abs_alpha_j(j + 1, alpha)) \
          * (alpha_j + lam)
        denom = sp.poch(2 * lam, alpha_j) \
          * sp.poch((D - j) / 2., abs_alpha_j(j + 1, alpha)) \
          * lam
        ca[di] = num / denom
    return np.sqrt(b * np.prod(ca))

def Y_cart(x, alpha):
    '''
    Spherical harmonic for cartesian argument
    Assumes 0 <= k <= d-2

    Dai & Xu
    '''
    d = len(x)
    r = 1
    prodarr = np.zeros(d - 2)
    for j in range(d - 2): # for d = 3, this is only one iteration
        t = cos_theta_k(x, d - (j + 1))
        prodarr[j] = sin_theta_k(x, d - (j + 1)) ** alpha_sum(j+1, alpha) \
          * gegenbauer(t, lambda_j(j, alpha), alpha[j])
    out = r ** np.sum(alpha) * g_alpha(alpha) * np.prod(prodarr)
    return out

def Y_ang(thetaa, alpha):
    '''
    Spherical harmonic for angular argument

    Dunkl & Xu, Theorem 4.1.4

    Parameters
    ----------
    thetaa : np.ndarray (D - 1, P)
    alpha:   np.ndarray (D, )

    Returns
    -------
    result : np.ndarray (P, )
    '''
    r = 1.
    D = thetaa.shape[0] + 1 # input dimension
    P = thetaa.shape[1]     # data points
    assert D > 2, "D > 2 required"
    assert alpha[-1] == 0 or alpha[-1] == 1, "alpha[-1] should be in {0, 1}"

    # form the entries in the product
    prodarr = np.zeros([D - 2, P])
    for di in range(D - 2):
        j = di + 1
        alpha_j = alpha[di]
        ta = thetaa[-j, :] # \theta_{d - j}
        lam = lambda_j(j, alpha)
        prodarr[di, :] = np.sin(ta) ** abs_alpha_j(j + 1, alpha) \
          * gegenbauer(np.cos(ta), lam, alpha_j)

    result = normalizing_factor(alpha) * r ** np.sum(alpha) \
      * g_alpha(alpha, thetaa[0, :]) * np.prod(prodarr, axis=0)
    return result.flatten()


def create_sphfeatures_matrix(theta, Nk):
    '''
    Create a matrix of spherical harmonics, including all the N(D,k) degenerate
    modes for each frequency k
    
    Parameters
    ==========
    theta : (D-1, P) angular coordinates to evaluate at
    Nk : maximum frequency

    Returns
    =======
    Y_mat : (N_alpha, P) evaluations of the spherical harmonics at P points
    k_arr : (N_alpha,) values of k
    '''
    D = theta.shape[0] + 1
    P = theta.shape[1]
    N_alpha = 0
    for k in range(Nk):
        N_alpha += num_harmonics(D,k)
    
    Y_mat = np.zeros([N_alpha, P])
    k_arr = np.zeros(N_alpha,int)
    
    na = 0 #total number of alpha vectors we've generated so far 
    for k in range(Nk): 
        #generate the N(D,k) alpha vectors 
        inds = list(itertools.combinations_with_replacement(range(D), k)) #all possible ways to put k balls into D buckets
        alphas = [np.bincount(ind, minlength=D) for ind in inds]
        alphas = [alpha for alpha in alphas if(alpha[-1] < 2)] #list of arrays, each an alpha vector
    
        # print("k: ", k, ", #harmonics: ", len(alphas),
        #           ", verifying with N(D,k): ", int(round(num_harmonics(D,k))),
        #           sep="")
        
        #Evaluate each spherical harmonic indexed by vector alpha, at all data points
        for ai, alpha in enumerate(alphas):
            Y = Y_ang(theta, alpha)
            Y_mat[na + ai, :] = Y #Put the Y vectors in a matrix
            k_arr[na + ai] = k
        
        na += num_harmonics(D,k)
    
    return Y_mat, k_arr
    
