# -*- coding: utf-8 -*-
"""
Created on Fri Feb 23 14:08:10 2018

@author: r.dewinter
"""

import numpy as np

def C3DTLZ1(x):
    gx = 100*(5 + np.sum( (x[1:]-0.5)**2 - np.cos(20*np.pi*(x[1:]-0.5))) )

    f1 = 0.5*(x[0])*(1+gx)
    f2 = 0.5*(1-x[0])*(1+gx)
    
    c1 = (f1**2)/4 + f2**2 - 1
    c2 = (f2**2)/4 + f1**2 - 1
    #-1* constr because of sacobra's constraint handling
    return [ np.array([f1, f2]), -1*np.array([c1,c2]) ]


#import matplotlib.pyplot as plt
#results = np.empty((1000000,2))
#constraints= np.empty((1000000,2))
#results[:] = np.nan
#constraints[:] = np.nan
#ii = 0
#for i in range(10):
#    for j in range(10):
#        for k in range(10):
#            for l in range(10):
#                for m in range(10):
#                    for n in range(10):
#                        x = np.array([i/10, j/10, k/10, l/10, m/10, n/10])
#                        results[ii], constraints[ii] = C3DTLZ1(x)
#                        ii+=1
#
#constr = np.sum(constraints<0, axis=1)==2
#results2 = results[constr]
#plt.plot(results2[:,0], results2[:,1], 'ro')
    
#par = len(rngMin)
#obj = len(ref)
#runs = 1000000
#objectives = np.empty((runs,obj))
#constraints = np.empty((runs,nconstraints))
#for i in range(runs):
#    x = rngMax*np.random.rand(par)+rngMin
#    objectives[i],constraints[i] = problemCall(x)
#np.sum(np.sum(constraints<0,axis=1)==nconstraints)/runs