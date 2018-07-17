# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 16:08:24 2018

@author: r.dewinter
"""
import numpy as np

def TNK(x):
    f1 = x[0]
    f2 = x[1]
    c1 = x[0]**2 + x[1]**2 - 1 - 0.1*np.cos(16*np.arctan(x[1]/x[0]))
    c2 = -1*((x[0]-0.5)**2 + (x[1]-0.5)**2 - 0.5)
    #-1* constr because of sacobra's constraint handling
    return [ np.array([f1, f2]), -1*np.array([c1,c2]) ]

#import matplotlib.pyplot as plt
#results = np.empty((1000000,2))
#constraints= np.empty((1000000,2))
#results[:] = np.nan
#constraints[:] = np.nan
#ii = 0
#for i in range(1000):
#    for j in range(1000):
#        x = np.array([i/1000*np.pi, j/1000*np.pi])
#        results[ii], constraints[ii] = TNK(x)
#        ii+=1
#
#constr = np.sum(constraints<0, axis=1)==2
#results2 = results[constr]
#plt.plot(results2[:,0], results2[:,1], 'ro')

#iteration time 236.3230001926422
#18381.99900007248
#18435.665999889374