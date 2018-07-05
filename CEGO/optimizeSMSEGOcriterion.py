# -*- coding: utf-8 -*-
"""
Created on Tue Oct 24 15:03:29 2017

@author: r.dewinter
"""
from predictorEGO import predictorEGO
from hypervolume import hypervolume
from paretofrontFeasible import paretofrontFeasible

import copy
import numpy as np
import time
import pygmo as pg

def optimizeSMSEGOcriterion(x, model, ref, paretoFront, currentHV, epsilon, gain):
    '''
    Slightly modified infill Criterion of the SMSEGO
    Ponweiser, Wagner et al. (Proc. 2008 PPSN, pp. 784-794)
    ***********************************************************************
    call: optimizeSMSEGOcriterion(x, model, ref, currentHV, eps, gain)
    
    arguments
    x:                  decision vector to be evaluated
    model:              d-dimensional cell array of models for each objective
    ref:                d-dimensional anti-ideal reference point
    paretoFront:        current Pareto front approximation
    currentHV:          hypervolume of current front with respect to ref
    epsilon:            epsilon to use in additive epsilon dominance
    gain:               gain factor for sigma (optional)
    '''  
#    print(x,'objectiv')
    nObj = len(model)
    mu = nObj*[None]
    mse = nObj*[None]
    for i in range(nObj):
        [mu[i], _, mse[i]] = predictorEGO(x, model[i])
    
    sigma = np.sqrt(mse)
    potentialSolution = np.ndarray.flatten(mu - gain*sigma)
    penalty = 0
    
    logicBool = np.all(paretoFront<= potentialSolution+epsilon, axis=1)
    for j in range(paretoFront.shape[0]):
        if logicBool[j]:
            p = - 1 + np.prod(1 + (potentialSolution-paretoFront[j,:]))
            penalty = max(penalty, p)
    if penalty == 0: #non-dominated solutions
        potentialFront = np.append(paretoFront, [potentialSolution], axis=0)
        myhv = hypervolume(potentialFront, ref)

        f = currentHV - myhv
    else:
        f = penalty
#    print(f,'objectiv')
    return f