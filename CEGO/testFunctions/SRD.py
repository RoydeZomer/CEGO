# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 10:25:06 2018

@author: r.dewinter
"""
import numpy as np

#def write_results(constraints, objectives):
#    try:
#        con = np.genfromtxt('conSRD.csv',delimiter=',')
#        obj = np.genfromtxt('objSRD.csv',delimiter=',')
#        con = np.asmatrix(con)
#        obj = np.asmatrix(obj)
#    except:
#        con = np.empty((0,11))
#        obj = np.empty((0,2))
#    con = np.append(con,constraints,axis=0)
#    obj = np.append(obj,objectives,axis=0)
#    np.savetxt('conSRD.csv',con,delimiter=',')
#    np.savetxt('objSRD.csv',obj,delimiter=',')


def SRD(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    fweight = 0.7854*(x1*(x2**2)*((10*(x3**2)/3)+(14.933*x3)-43.0934))-1.508*x1*((x6**2)+(x7**2))+7.477*((x6**3)+(x7**3))+0.7854*(x4*(x6**2)+x5*(x7**2))
    fstress = np.sqrt((((745*x4)/(x2*x3))**2)+(1.691E7))/(0.1*(x6**3))
    
    g1 = 1/(x1*(x2**2)*x3) - (1/27)
    g2 = 1/(x1*(x2**2)*(x3**2)) - (1/397.5)
    g3 = (x4**3)/(x2*x3*(x6**4)) - (1/1.93)
    g4 = (x5**3) / (x2*x3*(x7**4)) - (1/1.93)
    g5 = (x2*x3) - 40
    g6 = (x1/x2) - 12
    g7 = 5-(x1/x2)
    g8 = 1.9-x4+(1.5*x6)
    g9 = 1.9-x5+(1.1*x7)
    g10 = ((np.sqrt(((745*x4)/(x2*x3))**2) + 1.691**7 ) / (0.1*(x6**3)) ) - 1300
    g11 = ((np.sqrt(((745*x5)/(x2*x3))**2) + 1.5751**8 ) / (0.1*(x7**3)) ) - 1100
#    objectives = np.array([fweight, fstress])
#    constraints = np.array([g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11])
#    write_results([constraints],[objectives])
    return np.array([fweight, fstress]), np.array([g1,g2,g3,g4,g5,g6,g7,g8,g9,g10,g11])

#rngMin = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5])
#rngMax = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5])
#nVar = 7
#par = np.empty((1000000,7))
#objectives = np.empty((1000000,2))
#constraints = np.empty((1000000,11))
#objectives[:] = 0
#constraints[:] = 0
#for i in range(1000000):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    par[i] = x
#    obj, cons = SRD(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#a = np.sum(constraints<0, axis=1)==11
#b = paretofrontFeasible(objectives,np.array(len(objectives)*[[-1]]))
#plt.plot(objectives[b][:,0],objectives[b][:,1],'ro')
#objectives = np.append(objectives2,objectives[a][b],axis=0)
#iteration time 264.89999985694885
#23299.046999931335
#23353.990999937057
#iteration time 168.09800004959106
#22968.703000068665
#23029.43700003624
#iteration time 150.07700037956238
#22565.72100019455
#22633.31299972534
#iteration time 192.85300016403198
#17072.53400015831
#17122.828000068665