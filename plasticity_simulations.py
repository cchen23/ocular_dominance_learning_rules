# -*- coding: utf-8 -*-
"""
Created on Sun Dec 17 10:58:27 2017

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np
import sys

import plasticity_simulations_utils as utils

# TODO: Add averaging?
# TODO: Plot with multiple runs?
def basic_hebb_simulation():
    num_inputs = 2
    num_timesteps = 100
    all_inputs = utils.generate_inputs(num_inputs, num_timesteps)
    utils.run_simulation(utils.basic_hebb_update, all_inputs, "Basic Hebb")

def covariance_simulation():
    num_inputs = 2
    num_timesteps = 100
    all_inputs = utils.generate_inputs(num_inputs, num_timesteps)
    all_inputs -= np.reshape(np.mean(all_inputs, axis=1), (num_inputs, 1))
    utils.run_simulation(utils.basic_hebb_update, all_inputs, "Covariance")
    
def subtractive_normalization_simulation():
    num_inputs = 2
    num_timesteps = 1000
    all_inputs = utils.generate_inputs(num_inputs, num_timesteps)
    utils.run_simulation(utils.subtractive_normalization_update, all_inputs, "Subractive Normalization")

def oja_simulation():
    num_inputs = 2
    num_timesteps = 1000
    all_inputs = utils.generate_inputs(num_inputs, num_timesteps)
    utils.run_simulation(utils.oja_update, all_inputs, "Oja")
        
def main(argv):
    oja_simulation()
if __name__ == "__main__":
    main(sys.argv)