# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 11:06:16 2018

@author: r.dewinter
"""
import numpy as np

def DTLZ8(x):
    f1 = 1/3*np.sum(x[:3])
    f2 = 1/3*np.sum(x[3:6])
    f3 = 1/3*np.sum(x[6:])
    
    
    g1 = f3+4*f1-1
    g2 = f3+4*f2-1
    g3 = 2*f3+f1+f2-1
    
    return np.array([f1,f2, f3]), -1*np.array([g1,g2,g3])
    
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#rngMin = np.zeros(9)
#rngMax = np.ones(9)
#nVar = 9
#ref = np.array([1,1,1])
#parameters = np.empty((200,9))
#objectives = np.empty((200,3))
#constraints = np.empty((200,3))
#objectives[:] = 0
#constraints[:] = 0
#parameters[:]= 0
#for i in range(200):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    parameters[i] = x
#    obj, cons = DTLZ8(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#a = np.sum(constraints<0, axis=1)==3
##sum(a)
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
#ax.scatter(objectives[a][:,0], objectives[a][:,1], objectives[a][:,2])
#fig.show()