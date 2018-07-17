# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:44:51 2018

@author: r.dewinter
"""
import numpy as np

def SRN(x):
    f1 = 2 + (x[0]-2)**2 + (x[1]-1)**2
    f2 = 9*x[0] - (x[1]-1)**2
    c1 = x[0]**2 + x[1]**2 - 225
    c2 = x[0]-3*x[1]+10
    #-1* constr because of sacobra's constraint handling
    return [ np.array([f1, f2]), np.array([c1,c2]) ]

#import matplotlib.pyplot as plt
#results = np.empty((1000000,2))
#constraints= np.empty((1000000,2))
#results[:] = np.nan
#constraints[:] = np.nan
#ii = 0
#for i in range(1000):
#    for j in range(1000):
#        x = np.array([i/1000*40-20, j/1000*40-20])
#        results[ii], constraints[ii] = SRN(x)
#        ii+=1
#
#constr = np.sum(constraints<0, axis=1)==2
#results2 = results[constr]
#plt.plot(results2[:,0], results2[:,1], 'ro')
    
#iteration time 69.05299997329712
#10111.350999832153
#10159.71799993515
    
#iteration time 74.40999984741211
#5730.173999786377
#5777.526000022888
    
#import matplotlib.pyplot as plt
#
#rngMin = np.array([-20,-20])
#rngMax = np.array([20, 20])
#ref = np.array([301,72])
#nVar = 2
#nconstr = 2
#randomN = 100000
#parameters = np.empty((randomN,nVar))
#objectives = np.empty((randomN,2))
#constraints = np.empty((randomN,nconstr))
#objectives[:] = 0
#constraints[:] = 0
#parameters[:]= 0
#for i in range(randomN):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    parameters[i] = x
#    obj, cons = SRN(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
#for p in range(constraints.shape[1]):
#    x = parameters[:,0]
#    x2 = parameters[:,1]
#    z = constraints[:,p]
#        
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#        
#    ax.scatter(x, x2, z)