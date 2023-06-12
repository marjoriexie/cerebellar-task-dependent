# -*- coding: utf-8 -*-
"""
This file is part of the cerebellar-task-dependent repository (Xie et al. 2023)
Copyright (C) 2023 Marjorie Xie and Ashok Litwin-Kumar
See README for more information

Script that walks through how to set up our network model of the 
cerebellar mossy fiber - granule cell - Purkinje cell motif, 
generate targets from a Gaussian process, and train the model
to learn the target. 
You can run this script section by section.

"""

import numpy as np
import matplotlib.pyplot as plt
import networks
import targets 
import kernellib

plt.close('all')

# Choose network parameters 
D = 3 # dimension of input layer 
M = 20000 # number of granule cells 
f_desired = 0.3 # granule cell coding level 

# Set target parameters 
P = 30 # number of patterns used for training (smaller values will make learning harder)
Ptest = 100 # number of patterns used for testing 

#%% ****Part 1: model architecture, coding level, visualize network reprentations*********

# Generate some random patterns to feed into the input layer 
X = np.random.randn(D, P)
X = X / np.sqrt(np.sum(X ** 2, 0, keepdims=True)) # divide each vector by its length to restrict to unit sphere 

# Generate random mossy fiber-to-granule cell layer weights 
J = np.random.randn(M, D) * np.sqrt(1 / D) # scaling factor to ensure variance of current into a granule cell is 1 on average

# Generate population activity in the granule cell layer
# with a ReLU activation function applied to each granule cell.
# The same threshold will be applied to all granule cells.
H,_ = networks.relu( J @ X, f_desired)
    
# measure the coding level empirically 
f_empirical = np.mean(H > 0)
print('coding level = ' + str(np.round(f_empirical, 2)))

plt.figure()
plt.imshow(H[0:100, :])
plt.xlabel('Patterns')
plt.ylabel('Granule cells')
plt.title('Granule cell layer activity for different patterns \n f = ' + str(np.round(f_empirical, 2)))

#%% ****************** Part 2: Tasks  ***************************
# We will generate a family of target functons by sampling functions from a 
# Gaussian process. The length scale of a Gaussian process allows us to tune 
# how quickly the output varies as a function of the distance betwen two inputs. 

gamma = 1 # length scale of the GP
X, y = targets.draw_gp_functions_cholesky(gamma, D, P + Ptest, 
                                            normalize_target=True)

# Split the data into training and test sets
Xtrain = X[:, 0:P] # (D, P)
Xtest = X[:, P : P + Ptest] # (D, Ptest)
ytrain = y[:, 0:P] # (D, P)
ytest = y[:, P : P + Ptest] # (D, Ptest)

#%% *** Part 3: Measure the generalization performance of a network with a given coding level  *********
Htrain,_ = networks.relu( J @ Xtrain, f_desired)
w,_ = networks.get_regression_coefficients(Htrain, ytrain, lam=0)

# Use the test set to generate granule cell representations and outputs 
Htest, thresh = networks.relu( J @ Xtest, f_desired)
ytest_net = w.T @ Htest # network's output using the learned weights 

# Generalization error 
error = np.sum((ytest - ytest_net) ** 2) / np.sum(ytest ** 2)  

# Let's compare the performance to a baseline. We will compare it to a readout 
# directly from the task variables. 
w_baseline,_ = networks.get_regression_coefficients(Xtrain, ytrain, lam=0)
ytest_baseline = w_baseline.T @ Xtest
error_baseline = np.sum((ytest - ytest_baseline) ** 2) / np.sum(ytest ** 2)  

print('error = ' + str(error))
print('error baseline = ' + str(error_baseline))

if D == 1:
    fig, ax = plt.subplots()
    ax.set_title('Comparison of output of network with target')
    ax.plot(Xtest.flatten(), ytest.flatten(), 'r.', label='target')
    ax.plot(Xtest.flatten(), ytest_net.flatten(), 'k.', label='readout from expansion layer')
    ax.plot(Xtest.flatten(), ytest_baseline.flatten(), color='gray', label='readout from input layer')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
