# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 17:21:32 2018

@author: r.dewinter
"""
import numpy as np

def BNH(x):
    f1 = 4*x[0]**2+4*x[1]**2
    f2 = (x[0]-5)**2 + (x[1]-5)**2
    
    c1 = -1*((x[0]-5)**2 + x[1]-25)
    c2 = (x[0]-8)**2 + (x[1]-3)**2 - 7.7
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
#        x = np.array([i/1000*5, j/1000*3])
#        results[ii], constraints[ii] = BNH(x)
#        ii+=1
#
#constr = np.sum(constraints<0, axis=1)==2
#results2 = results[constr]
#plt.plot(results2[:,0], results2[:,1], 'ro')
    

#problemCall = BNH
#rngMin = np.array([0,0])
#rngMax = np.array([5,3])
#initEval = 30
#maxEval = 200
#smooth = 2
#runNo = 10
#ref = np.array([140,50])
#nconstraints = 2
#
#par = len(rngMin)
#runs = 1000000
#objectives = np.empty((runs,par))
#constraints = np.empty((runs,nconstraints))
#for i in range(runs):
#    x = rngMax*np.random.rand(par)+rngMin
#    objectives[i],constraints[i] = problemCall(x)
#np.sum(np.sum(constraints<0,axis=1)==nconstraints)/runs
    
#iteration time 96.12600016593933
#9458.411000013351
#9516.553999900818
    
#iteration time 34.580000162124634
#3196.130000114441
#3243.5099999904633
    
