# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 15:31:48 2018

@author: r.dewinter
"""

import numpy as np

def Kita(x):
    x1 = x[0]
    x2 = x[1]
    
    f1 = -1*(x2-x1**2)
    f2 = -1*(0.5*x1+x2+1)
    
    g1 = 6.5 - x1/6 -x2
    g2 = 7.5 - x1/2 -x2
    g3 = 30 - 5*x1 - x2
    
    return np.array([f1,f2]), -1*np.array([g1,g2,g3])
    
#import matplotlib.pyplot as plt
#rngMin = np.array([0, 0])
#rngMax = np.array([6, 6.5])
#nVar = 2
#ref = np.array([40,0])
#parameters = np.empty((1000000,2))
#objectives = np.empty((1000000,2))
#constraints = np.empty((1000000,3))
#objectives[:] = 0
#constraints[:] = 0
#parameters[:]= 0
#for i in range(1000000):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    parameters[i] = x
#    obj, cons = Kita(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#a = np.sum(constraints<0, axis=1)==3
##sum(a)
#
#plt.plot(objectives[a][:,0],objectives[a][:,1],'ro')
#plt.plot(ref[0],ref[1],'ro')