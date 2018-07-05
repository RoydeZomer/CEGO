# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 10:47:13 2018

@author: r.dewinter
"""
import numpy as np
#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter

from paretofrontFeasible import paretofrontFeasible

import pandas as pd
df = pd.read_csv("moga3200Orig.csv")

data = np.genfromtxt("moga_5100.csv", delimiter=',')
parameters = data[0:,1:7]
constraints = data[0:,7:-3]
objectives = data[0:,-3:-1]
constraints[:,:4] = constraints[:,:4]*-1+1
constraints[:,4:] = constraints[:,4:]*-1


#data = np.genfromtxt("withinResults.csv",delimiter=',')
#parameters = data[1:,:6]
#constraints = data[1:,6:-2]
#objectives = data[1:,-2:]


constraints = np.genfromtxt('results/compute_ship/run4_new/con_run4.csv', delimiter=',')
parameters = np.genfromtxt('results/compute_ship/run4_new/par_run4.csv', delimiter=',')
objectives = np.genfromtxt('results/compute_ship/run4_new/obj_run4.csv', delimiter=',')
visualizeResults(objectives, parameters, constraints)

#data = np.genfromtxt('moga2',delimiter='\t')

#objectives = data[:,-3:-1]
#parameters = data[:,1:7]
#feasible = data[:,0]+1
#feasible = np.nan_to_num(feasible)
#feasible = feasible<1
#
#constraints = data[:,0]+1
#constraints = np.asmatrix(constraints)
#constraints = constraints.T
#constraints = np.nan_to_num(constraints)
#import seaborn as sns
def visualizeResults(objectives, parameters, constraints):
    paretoOptimal = paretofrontFeasible(objectives,constraints)
    n_constraints = constraints.shape[1]
    feasible = np.sum(constraints <= 0, axis = 1) == n_constraints
    
#    sns.set_style('darkgrid')
    plt.plot(objectives[:,0], objectives[:,1], '.', c='r')
    plt.plot(objectives[feasible][:,0], objectives[feasible][:,1], 'o',c='b')
    
    paretoOptimalObjOrder = np.argsort(objectives[paretoOptimal][:,0], axis=0)
    paretoOptimalObj = objectives[paretoOptimal][paretoOptimalObjOrder]
    plt.plot(paretoOptimalObj[:,0], paretoOptimalObj[:,1], '^',c='g')
    plt.plot(objectives[0,0], objectives[0,1], 'D', c='gold')
    plt.plot(paretoOptimalObj[:,0], paretoOptimalObj[:,1], c='g')
    plt.title('paretofront')
    plt.show()
    
    objectives = objectives[feasible]
    parameters = parameters[feasible]
    constraints = constraints[feasible]
    
    feasible = np.sum(constraints <= 0, axis = 1) == n_constraints
    paretoOptimal = paretofrontFeasible(objectives,constraints)

    plt.plot(objectives[:,0], objectives[:,1], '.', c='r')
    plt.plot(objectives[feasible][:,0], objectives[feasible][:,1], 'o',c='b')
    
    paretoOptimalObjOrder = np.argsort(objectives[paretoOptimal][:,0], axis=0)
    paretoOptimalObj = objectives[paretoOptimal][paretoOptimalObjOrder]
    plt.plot(paretoOptimalObj[:,0], paretoOptimalObj[:,1], '^',c='g')
    plt.plot(objectives[0,0], objectives[0,1], 'D', c='gold')
    plt.plot(paretoOptimalObj[:,0], paretoOptimalObj[:,1], c='g')
    plt.title('paretofront')
    plt.show()
    
    plt.plot(paretoOptimalObj[:,0], paretoOptimalObj[:,1], '^',c='g')
    plt.show()
    
#    for p in range(parameters.shape[1]):
#        x = objectives[:,0]
#        y = objectives[:,1]
#        z = parameters[:,p]
#        
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        
#        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
#        
#    for c in range(constraints.shape[1]):
#        x = objectives[:,0]
#        y = objectives[:,1]
#        z = constraints[:,c]
#        
#        fig = plt.figure()
#        ax = fig.gca(projection='3d')
#        
#        ax.plot_trisurf(x, y, z, linewidth=0.2, antialiased=True)
