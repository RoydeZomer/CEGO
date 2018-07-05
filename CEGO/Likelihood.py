# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 17:03:55 2017

@author: r.dewinter
"""
import numpy as np
from regpoly0 import regpoly0
from corrnoisykriging import corrnoisykriging
from corrnoisygauss import corrnoisygauss
from dacefitEGO import dacefitEGO

def Likelihood(theta=None, parameters=None, objectives=None, nugget=None):
    m,n = parameters.shape
    thetaConv = np.power(10, theta[:n])
    if len(theta) >= 2*n and ( not(n==1 and len(theta)==2*n ) or nugget is not None) :
        #exponents are also optimized
        thetaConv = np.append(thetaConv, theta[n:2*n])
        if nugget is not None:
            thetaConv = np.append(thetaConv, nugget)
        else:
            thetaConv = np.append(thetaConv, theta[2*n])
        dmodel, perf = dacefitEGO(parameters, objectives, regpoly0, corrnoisykriging, thetaConv)
    else:
        #exponents are fixed to 2, only ativity parameters are optimized
        if nugget is not None:
            thetaConv = np.append(thetaConv,nugget)
        else:
            thetaConv = np.append(thetaConv,theta[n])
        dmodel, perf = dacefitEGO(parameters, objectives, regpoly0, corrnoisygauss, thetaConv)
    f = min(perf['perf'][len(thetaConv):])
    return f