# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:47:33 2017

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np

def run_simulation(update_method, all_inputs, update_name):
    num_inputs, num_timesteps = all_inputs.shape
    learning_rate = 0.1
    
    initial_weight = np.random.randn(num_inputs)
    all_weights = np.empty((num_inputs, num_timesteps + 1))
    all_weights[:,0] = initial_weight
    
    for i in range(num_timesteps):
        current_input = all_inputs[:, i]
        old_weight = all_weights[:,i]
        new_weight = update_method(old_weight, current_input, learning_rate)
        all_weights[:,i + 1] = new_weight
    
    # Plot results without averaging.
    plt.figure(0)
    plt.title("Weights from %s Update" % update_name)
    for i in range(num_inputs):
        weight_trajectory = all_weights[i,:]
        plt.plot(weight_trajectory, label="w%d" % i)
        plt.legend(loc="lower left")
        print("w%d: %f" % (i, weight_trajectory[-1]))

def generate_inputs(num_inputs, num_timesteps):
    all_inputs = np.random.rand(num_inputs, num_timesteps)
    return all_inputs # [0,1) so always non-negative.

def basic_hebb_update(weights, current_input, learning_rate):
    activation = np.dot(weights, current_input)
    activation = max(0, activation) # Prevent negative activity.
    return weights + learning_rate * activation * current_input

def subtractive_normalization_update(weights, current_input, learning_rate):
    activation = np.dot(weights, current_input)
    activation = max(0, activation) # Prevent negative activity.
    n = np.ones(current_input.shape)
    n[weights < 0] = 0
    n_u = np.sum(n)
    delta = activation * current_input - activation * np.dot(n, current_input) * n / n_u
    return weights + learning_rate * delta
    
def oja_update(weights, current_input, learning_rate):
    alpha = 0.1
    activation = np.dot(weights, current_input)
    activation = max(0, activation) # Prevent negative activity.
    delta = activation * current_input - alpha * activation * activation * weights
    return weights + learning_rate * delta