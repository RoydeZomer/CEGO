# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:15:02 2018

@author: r.dewinter
"""

import numpy as np

#def write_results(constraints, objectives):
#    try:
#        con = np.genfromtxt('conWB.csv',delimiter=',')
#        obj = np.genfromtxt('objWB.csv',delimiter=',')
#        con = np.asmatrix(con)
#        obj = np.asmatrix(obj)
#    except:
#        con = np.empty((0,5))
#        obj = np.empty((0,2))
#    con = np.append(con,constraints,axis=0)
#    obj = np.append(obj,objectives,axis=0)
#    np.savetxt('conWB.csv',con,delimiter=',')
#    np.savetxt('objWB.csv',obj,delimiter=',')

def WB(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    
    P = 6000
    L = 14
    E = 30*10**6
    G = 12*10**6
    tauMax = 13600
    sigmaMax = 30000
    
    Pc = ((4.013*E*np.sqrt( ((x3**2)*(x4**6))/36) ) / L**2 ) * (1-(x3/(2*L))*np.sqrt(E/(4*G)) )
    delta = (4*P*(L**3))/(E*x4*(x3)**3)
    M = P*(L+x2/2)
    R = np.sqrt( ((x2**2)/4) + ((x1+x3)/2)**2 )
    J = 2*(np.sqrt(2)*x1*x2*( (x2**2)/12 + ((x1+x3)/2)**2 )) 
    sigma = (6*P*L) / (x4*(x3**2))
    tauA = P/(np.sqrt(2)*x1*x2)
    tauAA = M*R / J
    tau = np.sqrt( tauA**2 + (2*tauA*tauAA*x2)/(2*R) + tauAA**2)
    
    f1 = 1.1047*(x1**2)*x2 + 0.04811*x3*x4*(14+x2)
    f2 = delta
    
    g1 = tau-tauMax
    g2 = sigma-sigmaMax
    g3 = x1-x4
    g4 = 0.125-x1
    g5 = P - Pc
    
#    write_results([np.array([g1,g2,g3,g4,g5])],[np.array([f1, f2])])
    
    return [np.array([f1, f2]), np.array([g1,g2,g3,g4,g5])]

#import matplotlib.pyplot as plt
#rngMin = np.array([0.125, 0.1, 0.1, 0.125])
#rngMax = np.array([5, 10, 10, 5])
#nVar = 4
#ref = np.array([350,0.1])
#parameters = np.empty((1000000,4))
#objectives = np.empty((1000000,2))
#constraints = np.empty((1000000,5))
#objectives[:] = 0
#constraints[:] = 0
#parameters[:]= 0
#for i in range(1000000):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    parameters[i] = x
#    obj, cons = WB(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#a = np.sum(constraints<0, axis=1)==5
##sum(a)
#
#plt.plot(objectives[a][:,0],objectives[a][:,1],'ro')
##plt.semilogy(350,1500, 'ro')


#iteration time 213.93099975585938
#20184.81399989128
#20227.919000148773