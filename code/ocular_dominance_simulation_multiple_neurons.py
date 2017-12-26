# -*- coding: utf-8 -*-
"""
Ocular dominance simulations with multiple neurons.

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np
import plasticity_simulations_utils as utils

# TODO: Put into jupyter notebook to make experimentation easier.
def recurrent_connections_simulation_averaged(K, update_function, update_name, save_name):
    num_neurons = 500
    num_timesteps = 1000
    num_inputs = 2
    learning_rate = 0.1
    weights = np.random.rand(num_neurons, num_inputs)
    
    Q = utils.generate_Q(num_inputs, num_timesteps)
    for timestep in range(num_timesteps):
        weights = update_function(weights, learning_rate, Q, K)
    
    weights_diff = np.reshape(weights[:,0] - weights[:,1], (num_neurons, 1))
    plt.figure()
    plt.title(r"Averaged Ocular Dominance Map, %s" % update_name)
    plt.imshow(np.transpose(weights_diff[:50,:]), cmap="gray")
    plt.savefig("../figures/map_averaged_%s" % save_name)
    plt.close()
    
def recurrent_connections_simulation(K, update_function, update_name, save_name):
    num_neurons = 500
    num_timesteps = 1000
    num_inputs = 2
    learning_rate = 0.1
    weights = np.random.rand(num_neurons, num_inputs)
    
    for timestep in range(num_timesteps):
        current_input = utils.generate_correlated_input()
        weights = update_function(weights, current_input, learning_rate, K)
    
    weights_diff = np.reshape(weights[:,0] - weights[:,1], (num_neurons, 1))
    plt.figure()
    plt.title(r"Ocular Dominance Map, %s" % update_name)
    plt.imshow(np.transpose(weights_diff[:50,:]), cmap="gray")
    plt.savefig("../figures/map_%s" % save_name)
    plt.close()
    
def recurrent_connections_simulation_competitivehebb(M, delta, update_function, update_name, save_name):
    num_neurons = 500
    num_timesteps = 1000
    num_inputs = 2
    learning_rate = 0.1
    weights = np.random.rand(num_neurons, num_inputs)
    
    for timestep in range(num_timesteps):
        current_input = utils.generate_correlated_input()
        weights = utils.competitive_hebb_update(weights, current_input, learning_rate, M, delta)
    
    weights_diff = np.reshape(weights[:,0] - weights[:,1], (num_neurons, 1))
    plt.figure()
    plt.title(r"Ocular Dominance Map, %s" % update_name)
    plt.imshow(np.transpose(weights_diff[:50,:]), cmap="gray")
    plt.savefig("../figures/map_%s" % save_name)
    plt.close()
    plt.acorr(weights_diff[:,0], maxlags=25)
    plt.title("autocorrelation %s." % update_name)
    plt.savefig("../figures/autocorrelation_%s" % save_name)
    plt.close()

def recurrent_connections_simulation_averaged_2D(K, update_function, update_name, save_name):
    num_neurons = 500
    num_timesteps = 1000
    num_inputs = 2
    learning_rate = 0.1
    weights = np.random.rand(num_neurons, num_inputs)
    num_rows = 10
    num_cols = 50
    
    Q = utils.generate_Q(num_inputs, num_timesteps)
    for timestep in range(num_timesteps):
        weights = update_function(weights, learning_rate, Q, K)
    
    weights_diff = np.reshape(weights[:,0] - weights[:,1], (num_neurons, 1))
    plt.figure()
    plt.title(r"Averaged Ocular Dominance Map, %s" % update_name)
    plt.imshow(np.reshape(weights_diff, (num_rows, num_cols)), cmap="gray")
    plt.savefig("../figures/map_2d_%s" % save_name)
    plt.close()

if __name__ == "__main__":
    num_neurons = 500
#    sigma = 2
#    K = utils.create_K_difference_Gaussians(num_neurons, sigma)
#    recurrent_connections_simulation_averaged(K, utils.basic_hebb_averaged_update, "Basic Hebb, $\sigma=%s$" % str(sigma), "basichebb_differencegaussians_sigma%s" % str(sigma).replace(".",""))
#    recurrent_connections_simulation(K, utils.basic_hebb_update, "Basic Hebb, $\sigma=%s$" % str(sigma), "basichebb_differencegaussians_sigma%s" % str(sigma).replace(".",""))
#    recurrent_connections_simulation(K, utils.subtractive_normalization_update_multiple, "Subtractive Normalization, $\sigma=%s$" % str(sigma), "subtractivenormalization_differencegaussians_sigma%s" % str(sigma).replace(".",""))

#    sigma = 0.5
#    K = utils.create_K_Gaussian(num_neurons, sigma)
#    recurrent_connections_simulation_averaged(K, utils.basic_hebb_averaged_update, "Basic Hebb, $\sigma=%s$" % str(sigma), "basichebb_gaussian_sigma%s" % str(sigma).replace(".",""))
#    recurrent_connections_simulation(K, utils.basic_hebb_update, "Basic Hebb, $\sigma=%s$" % str(sigma), "basichebb_gaussian_sigma%s" % str(sigma).replace(".",""))
#    recurrent_connections_simulation(K, utils.subtractive_normalization_update_multiple, "Subtractive Normalization, $\sigma=%s$" % str(sigma), "subtractivenormalization_gaussian_sigma%s" % str(sigma).replace(".",""))

# 
#    num_rows = 10
#    num_cols = 50
#    K = utils.create_K_2D_difference_Gaussians(num_rows, num_cols)
#    recurrent_connections_simulation_averaged_2D(K, utils.basic_hebb_averaged_update, "Basic Hebb", "basichebb_differencegaussians")

    sigma_e = 5
    sigma_i = 10
    delta = 10
    M = utils.create_M(num_neurons, sigma_e, sigma_i)
    recurrent_connections_simulation_competitivehebb(M, delta, utils.competitive_hebb_update, "Competitive Hebb, $\sigma_E=%s, \sigma_I=%s$" % (str(sigma_e), str(sigma_i)), "competitivehebb_sigmae%s_sigmai%s_delta%s" % (str(sigma_e).replace(".",""), str(sigma_i).replace(".",""), str(delta).replace(".","")))