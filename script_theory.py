# -*- coding: utf-8 -*-
"""
This file is part of the cerebellar-task-dependent repository (Xie et al. 2023)
Copyright (C) 2023 Marjorie Xie and Ashok Litwin-Kumar
See README for more information

Script that walks through how to compute generalization error analytically 
for a given kernel and Gaussian process target function. 
NOTE: The theory assumes D > 2. 

"""
import numpy as np
import matplotlib.pyplot as plt
import networks
import targets 
import kernellib
import sphlib
from scipy import integrate

plt.close('all')

#%% Network and target parameters

D = 3 # dimension of space of task variables 
Nk = 50 # include up to the Nk-th frequency term in the eigenfunction expansion
P = 30 # number of patterns used for training
Ptest = 100 # number of patterns in the test set 
gamma = 1 # length scale of Gaussian process
f_desired = 0.3 # coding level 

#%% 4.1 Compute eigenvalues of the kernel corresponding to the specified coding level
# (assuming a dot-product kernel and spherical harmonic basis)

kernel_func = kernellib.relu_kernel_intquad  # analytic expression for relu dot-product kernel
thresh = networks.f_to_thresh(f_desired)
meansub = False # mean-subtract the activations when computing the kernel
args = [thresh, meansub]

kernel_eig_arr = sphlib.eigenvalue_spectrum(kernel_func, args, D, Nk) # Nk-length array of eigenvalues

#plot the eigenvalues
fig, ax = plt.subplots()
ax.plot(np.arange(Nk), kernel_eig_arr, '.-', linewidth=1, markersize=2)
ax.semilogy()
xticks = np.arange(Nk, step=1)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ax.set_title('Eigenvalues for kernel of coding level f = ' + str(np.round(f_desired, 2)) )
ax.set_xlabel('Frequency k')
ax.set_ylabel('Eigenvalue')


#%% 4.2 Decompose the covariance function of the GP in the spherical harmonic basis.
# The target function can be written as a sum of spherical harmonics with zero-mean
# Gaussian distributed coefficients, whose variance we compute.
coeffvars = np.zeros(Nk) # variance of coefficients (c^2 in the paper)
print('Computing variance of coefficients of the Gaussian process for frequency k up to Nk = ' + str(Nk) + ' \n')
for ki in range(Nk):
    print('k = ', ki)
    coeffvars[ki] = integrate.quad(sphlib.funk_hecke_integrand, -1, 1,
                                      args=(ki, D, kernellib.sqr_exp_kernel,
                                            [gamma]))[0]
fig, ax = plt.subplots()
ax.plot(np.arange(Nk), coeffvars, '.-', linewidth=1, markersize=2)
ax.semilogy()
xticks = np.arange(Nk, step=4)
ax.set_xticks(xticks)
ax.set_xticklabels(xticks)
ax.set_title('Variance of coefficients of GP of length scale = ' + str(gamma) )
ax.set_xlabel('Frequency k')
ax.set_ylabel('$c^2$')

#%% 4.3 Draw a function from the Gaussian process 
X, y, coeff = targets.draw_gp_functions_coeffvar(coeffvars, gamma, D, P + Ptest, Nk=Nk, N_out=1)

#%%  4.4 Compute the generalization error for that specific target function 
# using the kernel eigenvalues and GP target coefficients.

# We have already computed the eigenvalues of the kernel for each frequency k but we need to specify 
# the eigenvalue for each mode alpha = {k, m}. The eigenvalues for a given k are the 
# same across m, and there are N(D,k) degenerate modes for each value of k.
# So we will create a new array with each eigenvalue repeated N(D,k) times for 
# each frequency k.
 
Ndk = np.zeros(Nk)
for k in range(Nk):
    Ndk[k] = sphlib.num_harmonics(D, k)
Ndk = Ndk.astype(int)
kernel_eig_alpha = np.repeat(kernel_eig_arr, Ndk, axis=0) 

# normalize by the surface area of the unit sphere
kernel_eig_alpha_norm = kernel_eig_alpha / sphlib.sfc_area(D)

#  generalization 
error_theory,_,_,_,_ = kernellib.gen_err_Bordelon(coeff[:,0], kernel_eig_alpha, P, lam=0, noise_std=0)
print(f'error_theory = {error_theory}')
