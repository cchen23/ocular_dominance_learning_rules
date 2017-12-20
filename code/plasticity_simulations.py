# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:58:27 2017

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

import plasticity_simulations_utils as utils

def run_simulation(update_method, all_inputs, update_name, nonnegative_weights):
    num_inputs, num_timesteps = all_inputs.shape
    learning_rate = 0.1
    
    initial_weight = np.random.randn(num_inputs)
    all_weights = np.empty((num_inputs, num_timesteps + 1))
    all_weights[:,0] = initial_weight
    
    for i in range(num_timesteps):
        current_input = all_inputs[:, i]
        old_weight = all_weights[:,i]
        new_weight = update_method(old_weight, current_input, learning_rate)
        if nonnegative_weights:
            new_weight[new_weight < 0] = 0
        all_weights[:,i + 1] = new_weight
    
    # Plot results without averaging.
    plt.figure(0)
    plt.title("Weights from %s Update" % update_name)
    for i in range(num_inputs):
        weight_trajectory = all_weights[i,:]
        plt.plot(weight_trajectory, label="w%d" % i)
        plt.legend(loc="lower left")
        print("w%d: %f" % (i, weight_trajectory[-1]))

# TODO: Add averaging?
# TODO: Plot with multiple runs?
def basic_hebb_simulation(nonnegative_weights):
    num_inputs = 2
    num_timesteps = 1000
    all_inputs = utils.generate_inputs(num_inputs, num_timesteps)
    run_simulation(utils.basic_hebb_update, all_inputs, "Basic Hebb", nonnegative_weights)

def covariance_simulation(nonnegative_weights):
    num_inputs = 2
    num_timesteps = 1000
    all_inputs = utils.generate_inputs(num_inputs, num_timesteps)
    all_inputs -= np.reshape(np.mean(all_inputs, axis=1), (num_inputs, 1))
    run_simulation(utils.basic_hebb_update, all_inputs, "Covariance", nonnegative_weights)
    
def subtractive_normalization_simulation(nonnegative_weights):
    num_inputs = 2
    num_timesteps = 1000
    all_inputs = utils.generate_inputs(num_inputs, num_timesteps)
    run_simulation(utils.subtractive_normalization_update, all_inputs, "Subractive Normalization", nonnegative_weights)

def oja_simulation(nonnegative_weights):
    num_inputs = 2
    num_timesteps = 1000
    all_inputs = utils.generate_inputs(num_inputs, num_timesteps)
    run_simulation(utils.oja_update, all_inputs, "Oja", nonnegative_weights)
        
if __name__ == "__main__":
    nonnegative_weights = True
    basic_hebb_simulation(nonnegative_weights)
