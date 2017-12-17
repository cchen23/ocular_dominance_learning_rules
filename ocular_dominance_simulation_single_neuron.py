# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 14:28:11 2017

@author: Cathy

Correlated inputs to both eyes. Only positive inputs, only positive weights.
"""
import matplotlib.pyplot as plt
import numpy as np
import plasticity_simulations_utils as utils

def run_average_simulation(num_inputs, num_timesteps, learning_rate, nonnegative_weights, update_function):
    initial_weight = np.abs(np.random.randn(num_inputs)) # Right, then left eye.
    all_weights = np.empty((num_inputs, num_timesteps + 1))
    all_weights[:,0] = initial_weight
    Q = utils.generate_Q(num_inputs, num_timesteps)
    
    for i in range(num_timesteps):
        weight = all_weights[:,i]
        new_weight = update_function(weight, learning_rate, Q)
        if nonnegative_weights:
            new_weight[new_weight < 0] = 0
        all_weights[:, i + 1] = new_weight
    return all_weights

def run_simulation_trajectory_average(update_function, update_name, nonnegative_weights=True):
    num_timesteps = 100
    num_inputs = 2
    learning_rate = 1
    all_weights = run_average_simulation(num_inputs, num_timesteps, learning_rate, nonnegative_weights, update_function)

    plt.figure()
    weights_trajectory_left = all_weights[0,:]
    weights_trajectory_right = all_weights[1,:]
    plt.plot(weights_trajectory_left, label="Left eye weights")
    plt.plot(weights_trajectory_right, label="Right eye weights")
    plt.xlabel("Number of Timesteps")
    plt.ylabel("Weight")
    plt.legend()
    plt.title("Ocular Dominance Weight Trajectories, Average %s Updates" % update_name)

def run_simulation_many_trials_average(update_function, update_name, nonnegative_weights=True):
    num_timesteps = 100
    num_inputs = 2
    learning_rate = 0.1
    num_trials = 10
    
    final_weights = np.empty((num_inputs, num_trials))
    for trial in range(num_trials):
        all_weights = run_average_simulation(num_inputs, num_timesteps, learning_rate, nonnegative_weights, update_function)
        final_weights[:, trial] = all_weights[:,-1]
    
    plt.figure()
    final_weights_left = final_weights[0,:]
    final_weights_right = final_weights[1,:]
    plt.scatter(final_weights_left, final_weights_right)
    plt.xlim([0, max(final_weights_left)])
    plt.ylim([0, max(final_weights_right)])
    plt.xlabel("Final Weights, Left Eye")
    plt.ylabel("Final Weights, Right Eye")
    plt.title("Ocular Dominance Final Weights, %s Updates" % update_name)

def run_simulation(num_inputs, num_timesteps, learning_rate, nonnegative_weights, update_function):
    initial_weight = np.abs(np.random.randn(num_inputs)) # Right, then left eye.
    all_weights = np.empty((num_inputs, num_timesteps + 1))
    all_weights[:,0] = initial_weight
    for i in range(num_timesteps):
        current_input = utils.generate_correlated_input()
        
        # Update weights.
        weights = all_weights[:,i]
        new_weights = update_function(weights, current_input, learning_rate)
        if nonnegative_weights:
            new_weights[new_weights < 0] = 0
        all_weights[:, i + 1] = new_weights
    return all_weights

# Weight trajectory with specified updates for a single eye.
def run_simulation_trajectory(update_function, update_name, nonnegative_weights=True):
    num_timesteps = 100
    num_inputs = 2
    learning_rate = 1
    all_weights = run_simulation(num_inputs, num_timesteps, learning_rate, nonnegative_weights, update_function)
        
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
        all_weights = run_simulation(num_inputs, num_timesteps, learning_rate, nonnegative_weights, update_function)
        final_weights[:, trial] = all_weights[:,-1]
    
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
    #run_simulation_many_trials(utils.basic_hebb_update, "Basic Hebb")
    #run_simulation_many_trials(utils.subtractive_normalization_update, "Subtractive Normalization")
    #run_simulation_many_trials(utils.oja_update, "Oja", nonnegative_weights=False)
    
    #run_simulation_trajectory(utils.basic_hebb_update, "Basic Hebb")
    #run_simulation_trajectory(utils.subtractive_normalization_update, "Subtractive Normalization")
    #run_simulation_trajectory(utils.oja_update, "Oja", nonnegative_weights=False)
    
    run_simulation_trajectory_average(utils.basic_hebb_averaged_update, "Basic Hebb")
    run_simulation_many_trials_average(utils.basic_hebb_averaged_update, "Basic Hebb")
