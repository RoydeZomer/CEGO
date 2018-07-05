# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 09:28:51 2018

@author: r.dewinter
"""
import numpy as np

def C1DTLZ1(x):
    gx = 100*(5 + np.sum( (x[1:]-0.5)**2 - np.cos(20*np.pi*(x[1:]-0.5))) )
    f1 = 0.5*(x[0])*(1+gx)
    f2 = 0.5*(1-x[0])*(1+gx)
    
    c = 1-(f2/0.6)-(f1/0.5)
    #-1* constr because of sacobra's constraint handling
    return [ np.array([f1, f2]), -1*np.array([c]) ]


#results = np.empty((1000000,2))
#constraints= np.empty((1000000,1))
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
#                        results[ii],constraints[ii] = C1DTLZ1(x)
#                        ii+=1
#
#constr = np.sum(constraints<0, axis=1)==1
#results2 = np.sum(results<1, axis=1)==2
#results2 = results[constr]    
#plt.plot(results2[:,0], results2[:,1], 'ro')