# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 11:47:33 2017

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np

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
    alpha = 1
    activation = np.dot(weights, current_input)
    activation = max(0, activation) # Prevent negative activity.
    delta = activation * current_input - alpha * activation * activation * weights
    return weights + learning_rate * delta