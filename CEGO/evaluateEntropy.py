# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 16:07:25 2017

@author: r.dewinter
"""
import numpy as np
import math

def evaluateEntropy(LHS, theta):
    n,m = LHS.shape
    if len(theta) < 2*m:
        theta = list(theta) + [2] * m
        nugget = 1
    elif len(theta) == 2 * m:
        nugget = 1
    else:
        nugget = theta[2*m]
    
    corrMatrix = np.empty((n,n))
    corrMatrix.fill(1)
    for k in range(n):
        for l in range(k+1,n):
            diff = theta[:m]*np.power(( abs( LHS[k,:] - LHS[l,:] )), theta[m:])
            corrMatrix[k,l] = nugget*math.exp(-sum(diff))
            corrMatrix[l,k] = corrMatrix[k,l]
    maxDet = max(np.linalg.det(corrMatrix), np.finfo(np.double).tiny)
    return -math.log10(maxDet)