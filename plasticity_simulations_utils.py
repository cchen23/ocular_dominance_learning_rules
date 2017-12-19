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

# TODO: investigate effect of different sigmas.
def create_K(num_neurons):
    # Sigma value from http://www.gatsby.ucl.ac.uk/~dayan/book/exercises/c8/c8.pdf
    K = np.empty((num_neurons, num_neurons))
    sigma = 0.66
    for i in range(num_neurons):
        for j in range(num_neurons):
            delta = (i - j)
            K[i,j] = np.exp(-(delta**2)/(2*sigma**2))-(1/9)*np.exp(-(delta**2)/(18*sigma**2))
    return K

# TODO: investigate effect of different sigmas.
def create_K_Gaussian(num_neurons):
    K = np.empty((num_neurons, num_neurons))
    sigma = 0.4
    for i in range(num_neurons):
        for j in range(num_neurons):
            delta = np.abs(i - j)
            K[i,j] = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp((-1/2) * (delta / sigma)**2)
    return K

def create_M(num_neurons):
    # "Mexican-hat" pattern with values defined in equation (2) in https://www.ncbi.nlm.nih.gov/pmc/articles/PMC2923481/
    M = np.empty((num_neurons, num_neurons))
    sigma_e = 0.05
    sigma_i = 0.2
    for i in range(num_neurons):
        for j in range(num_neurons):
            delta = (i - j)
            connection_e = 1/(sigma_e * np.sqrt(2 * np.pi)) * np.exp(-(delta ** 2) / (2 * sigma_e ** 2))
            connection_i = 1/(sigma_i * np.sqrt(2 * np.pi)) * np.exp(-(delta ** 2) / (2 * sigma_i ** 2))
            M[i,j] = connection_e - connection_i
    return M

def compute_activation(weights, current_input, K):
    if K is None:
        activation = np.dot(weights, current_input)
        activation = max(0, activation)
    else:
        activation = np.reshape(np.dot(K, np.dot(weights, current_input)), (K.shape[0], 1))
        activation[activation < 0] = 0
    return activation

def compute_activation_competitive_hebb(weights, current_input, M):
    delta = 5
    num_neurons = weights.shape[0]
    M = create_M(num_neurons)
    numerator = np.power(np.dot(weights, current_input), delta)
    z_a = numerator / np.sum(numerator)
    activation = np.dot(M, z_a)
    return np.expand_dims(activation, axis=1)

# Update rules.
def basic_hebb_update(weights, current_input, learning_rate, K=None):
    activation = compute_activation(weights, current_input, K)
    current_input = np.reshape(current_input, (1, current_input.shape[0]))
    return weights + learning_rate * np.dot(activation, current_input)

def basic_hebb_averaged_update(weight, learning_rate, Q, K=None):
    if K is None:
        K = np.eye(weight.shape[0])
    return weight + learning_rate * np.dot(K, np.dot(weight, Q))

# TODO: Update other rules with reshaped stuff to handle multiple neuron simulation.
def subtractive_normalization_update(weights, current_input, learning_rate, K=None):
    activation = compute_activation(weights, current_input, K)
    n = np.ones(current_input.shape)
    n[weights < 0] = 0
    n_u = np.sum(n)
    delta = activation * current_input - activation * np.dot(n, current_input) * n / n_u
    return weights + learning_rate * delta
    
def oja_update(weights, current_input, learning_rate, K=None):
    activation = compute_activation(weights, current_input, K)
    alpha = 1
    activation = max(0, activation) # Prevent negative activity.
    delta = activation * current_input - alpha * activation * activation * weights
    return weights + learning_rate * delta

# Update rules, multiple neurons.
def subtractive_normalization_update_multiple(weights, current_input, learning_rate, K):
    activation = compute_activation(weights, current_input, K)
    n = np.ones(current_input.shape)
    #n[weights < 0] = 0 # TODO: Add this back in?
    n_u = np.sum(n)
    delta = np.dot(activation, np.reshape(current_input, (1, 2))) - activation * np.dot(n, current_input) * n / n_u
    return weights + learning_rate * delta

def competitive_hebb_update(weights, current_input, learning_rate, M):
    activation = compute_activation_competitive_hebb(weights, current_input, M)
    current_input = np.reshape(current_input, (1, current_input.shape[0]))
    return weights + learning_rate * np.dot(activation, current_input)