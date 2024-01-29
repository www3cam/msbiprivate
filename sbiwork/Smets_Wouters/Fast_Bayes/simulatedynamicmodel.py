#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 13:11:20 2021

@author: cameron


"""

import numpy as np
import scipy as sc
from scipy.stats import norm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# alpha = 0.7
# delta = 0.15


#1. Set up grids for capital stock and \log z

#set up shock grid


# d = 2*(m*sigma/np.sqrt(1-rho**2))/(n_z - 1)
#discretize AR(1)
# z_grid = np.linspace(-m*sigma/np.sqrt(1-rho**2), m*sigma/np.sqrt(1-rho**2), n_z)
    
# #calculate transition matrix    
# trans_matrix = np.zeros([n_z, n_z])
# for i in range(n_z):
#     for j in range(n_z):
#         if j == 0:
#             trans_matrix[i,j] = norm.cdf((z_grid[0] + d/2 - rho*z_grid[i])/sigma)
#         elif j == n_z - 1:
#             trans_matrix[i,n_z - 1] = 1 - norm.cdf((z_grid[n_z - 1] - d/2 - rho*z_grid[i])/sigma)
#         else:
#             trans_matrix[i,j] = norm.cdf((z_grid[j] + d/2 - rho*z_grid[i])/sigma) - norm.cdf((z_grid[j] - d/2 - rho*z_grid[i])/sigma)
        

n_k = 81
n_z = 11
m = 3
# sigma = 0.2
# rho = 0.7
#pick some value for r
r = 0.05
    
#find steady state and set up geometric grid around it
def setup_k_grid(alpha, delta, z_grid, n_k):
    k_max = (np.exp(z_grid[-1])/delta)**(1/(1 - alpha))
    k_grid = np.zeros(n_k)
    for j in range(n_k):
        k_grid[n_k - j - 1] = k_max*(1 - delta)**j
    return k_grid



#do vfi using alpha and capital stock grid k_grid
def vfi(alpha, delta, sigma, rho):
    z_grid = np.zeros(n_z)
    k_grid = np.zeros(n_k)
    d = 2*(m*sigma/np.sqrt(1-rho**2))/(n_z - 1)
    #discretize AR(1)
    z_grid = np.linspace(-m*sigma/np.sqrt(1-rho**2), m*sigma/np.sqrt(1-rho**2), n_z)
    
    k_grid = setup_k_grid(alpha, delta, z_grid, n_k)
    

        
    #calculate transition matrix    
    trans_matrix = np.zeros([n_z, n_z])
    for i in range(n_z):
        for j in range(n_z):
            if j == 0:
                trans_matrix[i,j] = norm.cdf((z_grid[0] + d/2 - rho*z_grid[i])/sigma)
            elif j == n_z - 1:
                trans_matrix[i,n_z - 1] = 1 - norm.cdf((z_grid[n_z - 1] - d/2 - rho*z_grid[i])/sigma)
            else:
                trans_matrix[i,j] = norm.cdf((z_grid[j] + d/2 - rho*z_grid[i])/sigma) - norm.cdf((z_grid[j] - d/2 - rho*z_grid[i])/sigma)
                
    stacked_k = np.tile(k_grid, (n_k, 1)).T

    payoff_matr = np.zeros([n_k*n_z, n_k])
    k_theta = stacked_k**alpha
    for i in range(n_z):
        payoff_matr[i*n_k:(i + 1)*n_k,:] = np.exp(z_grid[i])*k_theta
        payoff_matr[i*n_k:(i + 1)*n_k,:] = payoff_matr[i*n_k:(i + 1)*n_k,:] - (stacked_k.T - (1 - delta)*stacked_k)

    tol = 1e-8
    err = 1
    v_old = np.zeros([n_z,n_k])
    v_new = np.ones([n_z, n_k])
    beta = 1/(1 + r)
    c = beta/(1-beta)
    policy = np.zeros([n_z, n_k]).astype(int)
    
    while err > tol:
        q = trans_matrix.dot(v_old)
        for i in range(n_z):
            hat_v = payoff_matr[i*n_k:(i+1)*n_k,:] + beta*q[i,:].T.reshape([n_k, 1]).dot(np.ones([1, n_k])).T

            v_new[i,:] = np.max(hat_v, axis = 1)
        #McQueen
        low_b = c*np.min(v_new - v_old)
        high_b = c*np.max(v_new - v_old)
        v_new = v_new + 0.5*(low_b + high_b)
        err = np.abs(np.max(v_new - v_old))

        v_old = v_new.copy()
        
    #get policy
    for i in range(n_z):
        hat_v = payoff_matr[i*n_k:(i+1)*n_k,:] + beta*q[i,:].T.reshape([n_k, 1]).dot(np.ones([1, n_k])).T
        policy[i,:] = np.argmax(hat_v, axis = 1)
        
    return v_new, policy




def simulate(s, n_drop, policy, v, sigma, rho):
    d = 2*(m*sigma/np.sqrt(1-rho**2))/(n_z - 1)
    z_grid = np.linspace(-m*sigma/np.sqrt(1-rho**2), m*sigma/np.sqrt(1-rho**2), n_z)
    trans_matrix = np.zeros([n_z, n_z])
    for i in range(n_z):
        for j in range(n_z):
            if j == 0:
                trans_matrix[i,j] = norm.cdf((z_grid[0] + d/2 - rho*z_grid[i])/sigma)
            elif j == n_z - 1:
                trans_matrix[i,n_z - 1] = 1 - norm.cdf((z_grid[n_z - 1] - d/2 - rho*z_grid[i])/sigma)
            else:
                trans_matrix[i,j] = norm.cdf((z_grid[j] + d/2 - rho*z_grid[i])/sigma) - norm.cdf((z_grid[j] - d/2 - rho*z_grid[i])/sigma)

    values = np.zeros(n_drop + s)
    #store indexes in k
    k = np.zeros(n_drop + s).astype(int)
    truek = np.zeros(n_drop + s).astype(float)
    #store indexes in z
    z = np.zeros(n_drop + s).astype(int)
    z_indexes = np.linspace(0, n_z -1, n_z).astype(int)

    k[0] = 0
    z[0] = 0
    values[0] = v[z[0], k[0]]

    for i in range(1,n_drop + s):
        z[i] = np.random.choice(a = z_indexes, p = trans_matrix[z[i-1],:])
        k[i] = policy[z[i], k[i-1]]
        truek[i] = k[i] +  + 3*np.random.randn()
        values[i] = v[z[i], k[i-1]]

    return truek[n_drop:], values[n_drop:]

def run_and_simulate(vectorin):
    s = 1000
    n_drop = 100
    val, pol = vfi(vectorin[0], vectorin[1], vectorin[2],vectorin[3])
    k, _ = simulate(s, n_drop, pol, val, vectorin[2], vectorin[3])
    return k

    

print(run_and_simulate([.7, .15, .2, .7]))
