# -*- coding: utf-8 -*-
"""
Create plots for written report.

@author: Cathy
"""
import matplotlib.pyplot as plt
import numpy as np

def get_K_Gaussian(x, sigma=0.4):
    y = 1/(sigma * np.sqrt(2 * np.pi)) * np.exp((-1/2) * (np.abs(x) / sigma)**2)
    return y

def get_M(x, sigma_e, sigma_i):
    connection_e = 10/(sigma_e * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2) / (2 * sigma_e ** 2))
    connection_i = 10/(sigma_i * np.sqrt(2 * np.pi)) * np.exp(-(x ** 2) / (2 * sigma_i ** 2))
    y = connection_e - connection_i
    return y

def get_K_difference_Gaussians(x, sigma=0.66):
    # Sigma value from http://www.gatsby.ucl.ac.uk/~dayan/book/exercises/c8/c8.pdf
    y = np.exp(-(x**2)/(2*sigma**2))-(1/9)*np.exp(-(x**2)/(18*sigma**2))
    return y

def create_plot(get_y_function, plot_title, save_name):
    x = np.linspace(-15,15,100) # 100 linearly spaced numbers
    y = get_y_function(x)
    plt.plot(x,y)
    plt.title(r'%s' % plot_title)
    plt.ylabel(r'$K[i,j]$')
    plt.xlabel(r'$i-j$')
    plt.savefig('../figures/%s' % save_name)
    plt.close()

def create_plot_many_sigmas():
    x = np.linspace(-25, 25,200)
    y1 = get_M(x, 10, 15)
    y2 = get_M(x, 10, 20)
    y3 = get_M(x, 5, 10)
    plt.plot(x, y1, label=r'$\sigma_e=10$, $\sigma_i=15$')
    plt.plot(x, y2, label=r'$\sigma_e=10$, $\sigma_i=20$')
    plt.plot(x, y3, label=r'$\sigma_e=5$, $\sigma_i=10$')
    lgd = plt.legend()
    plt.savefig('../figures/demo_mexicanhat', bbox_extra_artists=(lgd,), bbox_inches='tight')

if __name__ == "__main__":
#    create_plot(get_K_Gaussian, "Gaussian K, $\sigma=0.4$", "K_gaussian_sigma04")
#    create_plot(get_K_difference_Gaussians, "Difference of Gaussians K, $\sigma=0.66$", "K_difference_gaussian_sigma066")
#    create_plot(get_M, "M, $\sigma_E=0.05$, $\sigma_I=0.2$", "M_sigma00502")
    create_plot_many_sigmas()