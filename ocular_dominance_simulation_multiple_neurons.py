# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 16:39:58 2017

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np
import plasticity_simulations_utils as utils

def recurrent_connections_simulation():
    num_neurons = 500
    num_timesteps = 100
    num_inputs = 2
    learning_rate = 0.01
    weights = np.random.rand(num_neurons, num_inputs)
    
    Q = utils.generate_Q(num_inputs, num_timesteps)
    #K = utils.create_K(num_neurons)
    K = utils.create_K_Gaussian(num_neurons)
    for timestep in range(num_timesteps):
        weights = weights + learning_rate + np.dot(np.dot(K, weights), Q)
    
    weights_diff = np.reshape(weights[:,0] - weights[:,1], (num_neurons, 1))
    plt.imshow(np.transpose(weights_diff[:50,:]), cmap="gray")
    
if __name__ == "__main__":
    recurrent_connections_simulation()