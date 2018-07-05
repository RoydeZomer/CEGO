# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 15:05:43 2018

@author: r.dewinter
"""
import numpy as np

#def write_results(constraints, objectives):
#    try:
#        con = np.genfromtxt('conWP.csv',delimiter=',')
#        obj = np.genfromtxt('objWP.csv',delimiter=',')
#        con = np.asmatrix(con)
#        obj = np.asmatrix(obj)
#    except:
#        con = np.empty((0,7))
#        obj = np.empty((0,5))
#    con = np.append(con,constraints,axis=0)
#    obj = np.append(obj,objectives,axis=0)
#    np.savetxt('conWP.csv',con,delimiter=',')
#    np.savetxt('objWP.csv',obj,delimiter=',')


def WP(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    f1 = 106780.37 * (x2+x3) + 61704.67
    f2 = 3000 * x1
    f3 = 30570 * 0.02289 * x2 / (0.06*2289)**0.65
    f4 = 250 * 2289 * np.exp(-39.75*x2+9.9*x3 + 2.74)
    f5 = 25*((1.39/(x1*x2)) + 4940.0*x3 - 80.0)
    g1 = 0.00139/(x1*x2) + 4.94*x3 - 0.08 - 1
    g2 = 0.000306/(x1*x2)+1.082*x3 - 0.0986 - 1
    g3 = 12.307/(x1*x2) + 49408.24*x3 + 4051.02 - 50000
    g4 = 2.098/(x1*x2) + 8046.33*x3 - 696.71 - 16000
    g5 = 2.138/(x1*x2) + 7883.39*x3 - 705.04 - 10000
    g6 = 0.417*(x1*x2) + 1721.26*x3 - 136.54 - 2000
    g7 = 0.164/(x1*x2) + 631.13*x3 - 54.48 - 550
    
#    write_results([np.array([g1,g2,g3,g4,g5,g6,g7])],[np.array([f1,f2,f3,f4,f5])])
    
    return np.array([f1,f2,f3,f4,f5]), np.array([g1,g2,g3,g4,g5,g6,g7])


#rngMin = np.array([0.01,    0.01,  0.01])
#rngMax = np.array([0.45,    0.1,  0.1])
#nVar = 3
#nCon = 7
#runs = 1000000
#ref = np.array([1,1,1,1,1])
#parameters = np.empty((runs,nVar))
#objectives = np.empty((runs,len(ref)))
#constraints = np.empty((runs,nCon))
#objectives[:] = 0
#constraints[:] = 0
#parameters[:]= 0
#for i in range(runs):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    parameters[i] = x
#    obj, cons = WP(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#a = np.sum(constraints<0, axis=1)==nCon
#sum(a)
    
#iteration time 380.4010000228882
#39099.14999985695
#39152.895000219345
    
#iteration time 267.99399995803833
#23909.886000156403
#23952.8789999485