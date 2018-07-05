# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 14:12:43 2018

@author: r.dewinter
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Mar  2 12:15:02 2018

@author: r.dewinter
"""

import numpy as np

def DBD(x):
    x1=x[0]
    x2=x[1]
    x3=x[2]
    x4=x[3]
    
    f1 = 4.9*10**-5*(x2**2-x1**2)*(x4-1)
    f2 = ((9.82*(10**6))*(x2**2-x1**2))/(x3*x4*((x2**3)-(x1**3)))
    
    g1 = (x2-x1)-20
    g2 = 30 - 2.5*(x4+1)
    g3 = 0.4 - x3 / (3.14*((x2**2)-(x1**2)))
    g4 = 1- ( (2.22*(10**-3))*x3*(x2**3-x1**3) ) / (((x2**2) - (x1**2))**2)
    g5 = ((2.66*(10**-2)) * x3*x4*((x2**3)-(x1**3)))/((x2**2) - (x1**2)) - 900
    
    return [np.array([f1, f2]), -1*np.array([g1,g2,g3,g4,g5])]

#import matplotlib.pyplot as plt
#rngMin = np.array([55, 75, 1000, 2])
#rngMax = np.array([80, 110, 3000, 20])
#nVar = 4
#ref = np.array([5,50])
#parameters = np.empty((1000000,4))
#objectives = np.empty((1000000,2))
#constraints = np.empty((1000000,5))
#objectives[:] = 0
#constraints[:] = 0
#parameters[:]= 0
#for i in range(1000000):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    parameters[i] = x
#    obj, cons = DBD(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#a = np.sum(constraints<0, axis=1)==5
##sum(a)
#
#plt.plot(objectives[a][:,0],objectives[a][:,1],'ro')


#iteration time 170.3399999141693
#17684.232000112534
#17729.22600030899
    
#97390.70500040054
#97437.12800002098
    
#iteration time 111.31799983978271
#13988.68099975586
#14037.473000049591