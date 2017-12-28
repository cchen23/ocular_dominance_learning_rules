# ocular_dominance_learning_rules

This project simulates the development of ocular dominance preferences in visual cortex. The code to produce simulations are in the code/ folder.
Within this folder, the code is organized as follows:

## plasticity_simulations_utils.py
This contains definitions of learning rules, recurrent connections, and input generation. These modules can be swapped into the simulations for ocular dominance.

## ocular_dominance_simulation_single_neuron.py
This simulates the development of ocular dominance in a single neuron. It plots the trajectories of weight learning in a single trial, or the final weights learned in multiple trials. By specifying a learning rule defined in plasticity_simulations_utils.py, this can simulate the effect of various learning rules on ocular dominance development.

## ocular_dominance_simulation_multiple_neurons.py
This simulates the development of ocular dominance maps in a network of neurons. It plots the resulting ocular dominance maps (in terms of the difference between each neuron's left and right eye preferences). By specifying a learning rule and recurrent connection defined in plasticity_simulations_utils.py, this can simulate the effect of various learning rules and recurrent connections on ocular dominance map development.
