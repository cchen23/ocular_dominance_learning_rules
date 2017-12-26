# -*- coding: utf-8 -*-
"""
Utility functions for ocular dominance simulations.

@author: Cathy
"""
import numpy as np

# Input generation methods.
def generate_inputs(num_inputs, num_timesteps):
    all_inputs = np.random.rand(num_inputs, num_timesteps)
    return all_inputs # [0,1) so always non-negative.

def generate_correlated_input():
    shared_input = np.abs(np.random.randn())
    input_left = shared_input + 0.5 * np.random.randn()
    input_right = shared_input + 0.5 * np.random.randn()
    return np.array([input_left, input_right])

def generate_Q(num_inputs, num_timesteps):
    Q = np.zeros((num_inputs, num_inputs))
    for i in range(num_timesteps):
        shared_input = np.abs(np.random.randn())
        input_left = shared_input + 0.1 * np.random.randn()
        input_right = shared_input + 0.1 * np.random.randn()
        Q[0,0] += input_right * input_right
        Q[1,1] += input_left * input_left
        Q[0,1] += input_left * input_right
        Q[1,0] += input_left * input_right
    Q = Q / num_timesteps
    return Q

# Recurrent connection methods.
def create_K_difference_Gaussians(num_neurons, sigma):
    # Sigma value from http://www.gatsby.ucl.ac.uk/~dayan/book/exercises/c8/c8.pdf
    K = np.empty((num_neurons, num_neurons))
    for i in range(num_neurons):
        for j in range(num_neurons):
            delta = (i - j)
            K[i,j] = np.exp(-(delta**2)/(2*sigma**2))-(1/9)*np.exp(-(delta**2)/(18*sigma**2))
    return K

def index_to_location(index, num_rows):
    row = index / num_rows
    col = index % num_rows
    return row, col

def create_K_2D_difference_Gaussians(num_rows, num_cols):
    # Sigma value from http://www.gatsby.ucl.ac.uk/~dayan/book/exercises/c8/c8.pdf
    num_neurons = num_rows * num_cols
    K = np.empty((num_neurons, num_neurons))
    sigma = 0.66
    for i in range(num_neurons):
        for j in range(num_neurons):
            row_i, col_i = index_to_location(i, num_rows)
            row_j, col_j = index_to_location(j, num_rows)
            delta = np.abs(row_i-row_j) + np.abs(col_i-col_j)
            K[i,j] = np.exp(-(delta**2)/(2*sigma**2))-(1/9)*np.exp(-(delta**2)/(18*sigma**2))
    return K
 
def create_K_Gaussian(num_neurons, sigma):
    K = np.empty((num_neurons, num_neurons))
    for i in range(num_neurons):
        for j in range(num_neurons):
            delta = np.abs(i - j)
            K[i,j] = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp((-1/2) * (delta / sigma)**2)
    return K

def create_M(num_neurons, sigma_e, sigma_i):
    # "Mexican-hat" pattern with values defined in equation (2) in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2923481/
    M = np.empty((num_neurons, num_neurons))
    for i in range(num_neurons):
        for j in range(num_neurons):
            delta = (i - j)
            connection_e = 10/(sigma_e * np.sqrt(2 * np.pi)) * np.exp(-(delta ** 2) / (2 * sigma_e ** 2))
            connection_i = 10/(sigma_i * np.sqrt(2 * np.pi)) * np.exp(-(delta ** 2) / (2 * sigma_i ** 2))
            M[i,j] = connection_e - connection_i
    return M

# Activation methods.
def eliminate_edge_effects(activation):
    activation_edge = (activation[0] + activation[-1]) / 2 # Eliminate edge effects
    activation[0] = activation_edge
    activation[-1] = activation_edge
    return activation

def compute_activation(weights, current_input, K):
    if K is None:
        activation = np.dot(weights, current_input)
        activation = max(0, activation)
    else:
        activation = np.reshape(np.dot(K, np.dot(weights, current_input)), (K.shape[0], 1))
        activation[activation < 0] = 0
        activation = eliminate_edge_effects(activation)
    return activation

def compute_activation_competitive_hebb(weights, current_input, M, delta):
    numerator = np.power(np.dot(weights, current_input), delta)
    z_a = numerator / np.sum(numerator)
    activation = np.dot(M, z_a)
    activation = eliminate_edge_effects(activation)
    activation[activation < 0] = 0
    return np.expand_dims(activation, axis=1)

# Update rule definitions.
def basic_hebb_update(weights, current_input, learning_rate, K=None):
    activation = compute_activation(weights, current_input, K)
    current_input = np.reshape(current_input, (1, current_input.shape[0]))
    return weights + learning_rate * np.dot(activation, current_input)

def basic_hebb_averaged_update(weight, learning_rate, Q, K=None):
    if K is None:
        K = np.eye(weight.shape[0])
    return weight + learning_rate * np.dot(K, np.dot(weight, Q))

def subtractive_normalization_update(weights, current_input, learning_rate, K=None):
    activation = compute_activation(weights, current_input, K)
    n = np.ones(current_input.shape)
    n_u = np.sum(n)
    delta = activation * current_input - activation * np.dot(n, current_input) * n / n_u
    weights = weights + learning_rate * delta
    weights[weights < 0] = 0
    return weights
    
def oja_update(weights, current_input, learning_rate, K=None):
    activation = compute_activation(weights, current_input, K)
    alpha = 1
    activation = max(0, activation) # Prevent negative activity.
    delta = activation * current_input - alpha * activation * activation * weights
    return weights + learning_rate * delta

# Update rule definitions, multiple neurons.
def subtractive_normalization_update_multiple(weights, current_input, learning_rate, K):
    activation = compute_activation(weights, current_input, K)
    n = np.ones(current_input.shape)
    #n[weights < 0] = 0 # TODO: Add this back in?
    n_u = np.sum(n)
    delta = np.dot(activation, np.reshape(current_input, (1, 2))) - activation * np.dot(n, current_input) * n / n_u
    return weights + learning_rate * delta

def competitive_hebb_update(weights, current_input, learning_rate, M, delta):
    activation = compute_activation_competitive_hebb(weights, current_input, M, delta)
    current_input = np.reshape(current_input, (1, current_input.shape[0]))
    return weights + learning_rate * np.dot(activation, current_input)
