# -*- coding: utf-8 -*-
"""
Created on Wed Mar  7 14:12:18 2018

@author: r.dewinter
"""
import numpy as np

def CSI(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    x7 = x[6]
    f1 = 1.98 + 4.9*x1 + 6.67*x2 + 6.98*x3 + 4.01*x4 + 1.78*x5 + 0.00001*x6 + 2.73*x7
    F = 4.72 - 0.5*x4 - 0.19*x2*x3
    f2 = F
    Vmbp = 10.58 - 0.674*x1*x2 - 0.67275*x2
    Vfd = 16.45 - 0.489*x3*x7 - 0.843*x5*x6
    f3 = 0.5*(Vmbp+Vfd)
    
    g1 = 1.16 - 0.3717*x2*x4 - 0.0092928*x3 - 1
    g2 = 0.261 - 0.0159*x1*x2 - 0.06486*x1 - 0.019*x2*x7 + 0.0144*x3*x5 + 0.0154464*x6 - 0.32
    g3 = 0.214 + 0.00817*x5 - 0.045195*x1 - 0.0135168*x1 + 0.03099*x2*x6 - 0.018*x2*x7 + 0.007176*x3 + 0.023232*x3 - 0.00364*x5*x6 - 0.018*(x2**2) - 0.32
    g4 = 0.74 - 0.61*x2 - 0.031296*x3 - 0.031872*x7 + 0.227*(x2**2) - 0.32
    g5 = 28.98 + 3.818*x3 - 4.2*x1*x2 + 1.27296*x6 - 2.68065*x7-32
    g6 = 33.86 + 2.95*x3 - 5.057*x1*x2 - 3.795*x2 - 3.4431*x7 + 1.45728 - 32
    g7 = 46.36 - 9.9*x2 - 4.4505*x1 - 32
    g8 = F - 4
    g9 = Vmbp - 9.9
    g10 = Vfd - 15.7
    
    return np.array([f1,f2,f3]), np.array([g1,g2,g3,g4,g5,g6,g7,g8,g9,g10])
    
#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from paretofrontFeasible import paretofrontFeasible
#rngMin = np.array([0.5,    0.45,  0.5,  0.5,   0.875,     0.4,    0.4])
#rngMax = np.array([1.5,    1.35,  1.5,  1.5,   2.625,     1.2,    1.2])
#nVar = 7
#nconstr = 10
#ref = np.array([42,4.5,13])
#randomN = 100
#parameters = np.empty((randomN,nVar))
#objectives = np.empty((randomN,3))
#constraints = np.empty((randomN,nconstr))
#objectives[:] = 0
#constraints[:] = 0
#parameters[:]= 0
#for i in range(randomN):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    parameters[i] = x
#    obj, cons = CSI(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#a = np.sum(constraints<0, axis=1)==nconstr
##sum(a)
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
##ax.scatter(objectives[a][:,0], objectives[a][:,1], objectives[a][:,2])
#b = paretofrontFeasible(objectives[a],np.zeros(objectives[a].shape))
#ax.scatter(objectives[a][b][:,0], objectives[a][b][:,1], objectives[a][b][:,2])
#fig.show()
    
#iteration time 287.0810000896454
#22689.661999940872
#22737.164000034332
    
#iteration time 234.7699999809265
#22036.71199989319
#22086.0640001297
    
#iteration time 215.57699990272522
#21837.662999629974
#21889.319999933243
    
#iteration time 216.04899978637695
#22516.34399986267
#22566.427999973297