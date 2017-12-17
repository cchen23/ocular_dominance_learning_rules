# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:28:11 2017

@author: Cathy

Correlated inputs to both eyes. Only positive inputs, only positive weights.
"""
import matplotlib.pyplot as plt
import numpy as np
import plasticity_simulations_utils as utils

def run_simulation_average():
    num_timesteps = 100
    num_inputs = 2
    learning_rate = 1
    initial_weight = np.abs(np.random.randn(num_inputs)) # Right, then left eye.
    #all_weights = np.empty((num_inputs, num_timesteps + 1))
    #all_weights[:,0] = initial_weight
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
    weight = initial_weight + learning_rate * np.dot(Q, initial_weight)
    print(weight)
    print("hi")
    print(np.dot(Q, initial_weight))

# Weight trajectory with specified updates for a single eye.
def run_simulation_trajectory(update_function, update_name, nonnegative_weights=True):
    num_timesteps = 100
    num_inputs = 2
    learning_rate = 1
    initial_weight = np.abs(np.random.randn(num_inputs)) # Right, then left eye.
    all_weights = np.empty((num_inputs, num_timesteps + 1))
    all_weights[:,0] = initial_weight
    for i in range(num_timesteps):
        # Generate correlated inputs.
        shared_input = np.abs(np.random.randn())
        input_left = shared_input + 0.5 * np.random.randn()
        input_right = shared_input + 0.5 * np.random.randn()
        current_input = np.array([input_left, input_right])
        
        # Update weights.
        weights = all_weights[:,i]
        new_weights = update_function(weights, current_input, learning_rate)
        if nonnegative_weights:
            new_weights[new_weights < 0] = 0
        all_weights[:, i + 1] = new_weights
        
    plt.figure()
    weights_trajectory_left = all_weights[0,:]
    weights_trajectory_right = all_weights[1,:]
    plt.plot(weights_trajectory_left, label="Left eye weights")
    plt.plot(weights_trajectory_right, label="Right eye weights")
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Weight")
    plt.legend()
    plt.title("Ocular Dominance Weight Trajectories, %s Updates" % update_name)

# Final weights from multiple trials with specified updates.
def run_simulation_many_trials(update_function, update_name, nonnegative_weights=True):
    print(update_name)
    num_timesteps = 100
    num_inputs = 2
    learning_rate = 0.1
    num_trials = 10
    
    final_weights = np.empty((num_inputs, num_trials))
    for trial in range(num_trials):
        initial_weight = np.abs(np.random.randn(num_inputs)) # Right, then left eye.
        all_weights = np.empty((num_inputs, num_timesteps + 1))
        all_weights[:,0] = initial_weight
        for timestep in range(num_timesteps):
            # Generate correlated inputs.
            shared_input = np.abs(np.random.randn())
            input_left = shared_input + 0.5 * np.random.randn()
            input_right = shared_input + 0.5 * np.random.randn()
            current_input = np.array([input_left, input_right])
            
            # Update weights.
            weights = all_weights[:,timestep]
            new_weights = update_function(weights, current_input, learning_rate)
            if nonnegative_weights:
                new_weights[new_weights < 0] = 0
            all_weights[:, timestep + 1] = new_weights
        final_weights[:, trial] = new_weights
        print(new_weights)
    
    plt.figure()
    final_weights_left = final_weights[0,:]
    final_weights_right = final_weights[1,:]
    plt.scatter(final_weights_left, final_weights_right)
    plt.xlim([0, max(final_weights_left)])
    plt.ylim([0, max(final_weights_right)])
    plt.xlabel("Final Weights, Left Eye")
    plt.ylabel("Final Weights, Right Eye")
    plt.title("Ocular Dominance Final Weights, %s Updates" % update_name)

# TODO: When to enforce nonnegative weights?
if __name__ == "__main__":
    run_simulation_many_trials(utils.basic_hebb_update, "Basic Hebb")
    run_simulation_many_trials(utils.subtractive_normalization_update, "Subtractive Normalization")
    run_simulation_many_trials(utils.oja_update, "Oja", nonnegative_weights=False)
    
    run_simulation_trajectory(utils.basic_hebb_update, "Basic Hebb")
    run_simulation_trajectory(utils.subtractive_normalization_update, "Subtractive Normalization")
    run_simulation_trajectory(utils.oja_update, "Oja", nonnegative_weights=False)
