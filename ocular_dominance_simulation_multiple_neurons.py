# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 16:39:58 2017

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np
import plasticity_simulations_utils as utils

# TODO: Create 2D simulation. Should be able to do it by making K matrix represent 2D Gaussian (using mod?) and reshaping weights array.
# TODO: Put into jupyter notebook to make experimentation easier.
def recurrent_connections_simulation_averaged(update_function, update_name):
    num_neurons = 500
    num_timesteps = 100
    num_inputs = 2
    learning_rate = 0.1
    weights = np.random.rand(num_neurons, num_inputs)
    
    Q = utils.generate_Q(num_inputs, num_timesteps)
    #K = utils.create_K(num_neurons)
    K = utils.create_K_Gaussian(num_neurons)
    for timestep in range(num_timesteps):
        weights = update_function(weights, learning_rate, Q, K)
    
    weights_diff = np.reshape(weights[:,0] - weights[:,1], (num_neurons, 1))
    plt.figure()
    plt.title("Averaged Ocular Dominance Map, %s Update" % update_name)
    plt.imshow(np.transpose(weights_diff[:50,:]), cmap="gray")
    
def recurrent_connections_simulation(update_function, update_name):
    num_neurons = 500
    num_timesteps = 100
    num_inputs = 2
    learning_rate = 0.1
    weights = np.random.rand(num_neurons, num_inputs)
    
    #K = utils.create_K(num_neurons)
    K = utils.create_K_Gaussian(num_neurons)
    for timestep in range(num_timesteps):
        current_input = utils.generate_correlated_input()
        weights = update_function(weights, current_input, learning_rate, K)
    
    weights_diff = np.reshape(weights[:,0] - weights[:,1], (num_neurons, 1))
    plt.figure()
    plt.title("Ocular Dominance Map, %s Update" % update_name)
    plt.imshow(np.transpose(weights_diff[:50,:]), cmap="gray")
    
if __name__ == "__main__":
    #recurrent_connections_simulation_averaged(utils.basic_hebb_averaged_update, "Basic Hebb")
    #recurrent_connections_simulation(utils.basic_hebb_update, "Basic Hebb")
    recurrent_connections_simulation(utils.subtractive_normalization_update_multiple, "Subtractive Normalization")