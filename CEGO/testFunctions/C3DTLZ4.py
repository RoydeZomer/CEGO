# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 16:50:49 2018

@author: r.dewinter
"""

import numpy as np

def C3DTLZ4(x):
    x = np.array(x)
    gx = np.sum((x[1:]-0.5)**2)
    
    
    f1 = (1+(gx))*np.cos(x[0]*np.pi/2)
    f2 = (1+(gx))*np.sin(x[0]*np.pi/2)
    
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
#                        results[ii], constraints[ii] = C3DTLZ4(x)
#                        ii+=1
#
#constr = np.sum(constraints<0, axis=1)==2
#results2 = np.sum(results<1, axis=1)==2
#results2 = results[constr]
#plt.plot(results2[:,0], results2[:,1], 'ro')
    
#iteration time 193.98799991607666
#21351.97400021553
#21406.330000162125