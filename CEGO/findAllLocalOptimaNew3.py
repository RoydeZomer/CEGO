# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 12:25:40 2017

@author: r.dewinter
"""
from computeStartPoints import computeStartPoints
from adaptiveFiltering import adaptiveFiltering
from RbfInter import predictConstraints

import time
import numpy as np
from scipy import optimize
from functools import partial

import multiprocessing

def minimizeCriteria(arguments):
    startPoints, maxEval, sLB, sUB, lb, ub, EPS, constraintSurrogates, criterion = arguments
    nStartingPoints, n = startPoints.shape
    
    evals = 0
    X = np.empty((nStartingPoints, n))
    X.fill(np.nan)
    Y = np.empty(nStartingPoints)
    Y.fill(np.nan)

    X2 = np.empty((nStartingPoints, n))
    X2.fill(np.nan) 
    Y2 = np.empty(nStartingPoints)
    Y2.fill(np.nan)
    
    X3 = np.empty((nStartingPoints, n))
    X3.fill(np.nan) 
    CV = np.empty(nStartingPoints)
    CV.fill(np.nan)
    
    i = 0    
#    partialConstraintPredictionsFunction = partial(predictConstraints2, rbfmodel=constraintSurrogates, EPS=EPS)
    partialConstraintPredictionsFunction = partial(predictConstraints, rbfmodel=constraintSurrogates, EPS=EPS)        
    
    while evals < maxEval and i < nStartingPoints:
        x0 = startPoints[i,:]
        cons = []
        #add constraints as function constraints
        cons.append({'type':'ineq','fun':partialConstraintPredictionsFunction})
        
        #add lower and upper bound
        for factor in range(len(sLB[i])):
            lower = sLB[i][factor]
            l = {'type':'ineq','fun': lambda x, lbb=lower, i=factor: x[i]-lbb}
            cons.append(l)
        for factor in range(len(sUB[i])):
            upper = sUB[i][factor]
            u = {'type':'ineq','fun': lambda x, ubb=upper, i=factor: ubb-x[i]}
            cons.append(u)
        
        opts = {'maxiter':500, 'tol':1e-6, 'catol':1e-6}
       
        optimResult = optimize.minimize(criterion, x0, constraints=cons, options=opts, method='COBYLA')

        Xtemp = optimResult.x
        Ytemp = optimResult.fun
        stopFlag = optimResult.success
        maxcv = optimResult.maxcv
        funcCount = optimResult.nfev
        evals += funcCount
        
        if (stopFlag): #if cobyla succesfully terminated this is probably the best option
            X[i,:] = np.maximum(np.minimum(Xtemp,ub),lb)
            Y[i] = Ytemp
        elif maxcv<=1e-6: # if xtemp has no violation
            X2[i,:] = np.maximum(np.minimum(Xtemp,ub),lb)
            Y2[i] = Ytemp
        else: #else if cobyla did not found a feasible minima save x with the expected constraint violation value
            X3[i,:] = np.maximum(np.minimum(Xtemp,ub),lb)
            CV[i] = maxcv
        i += 1
        if evals >= maxEval:
            print('max evals reached in FindAllLocalOptimaNew')
            
    return(X,Y,X2,Y2,X3,CV)



def findAllLocalOptimaNew(model1=None, lb=None, ub=None, criterion=None, constraintSurrogates=None, EPS=None, maxEval=50000, tol=1e-16):
    '''
    [X, Y] = findAllLocalOptimaNew(model1, lb, ub, criterion, maxEval, tol, makeDebugOutput)
    ----------------------------------------------------------------------
    Find all local optima of a specified model1 infill criterion using
    search-space-partion-based restarts of a gradient technique
    
    Call
    [X, Y] = findAllLocalOptimaNew(model1, lb, ub, obj, ...
       criterion, maxEval, tol, makeDebugOutput)
    
    Output arguments
    X: local optima of the specified model1 infill criterion
    Y: corresponding values of the infill criterion (optional)
    
    Input arguments
    model1: cell array of m DACE model1s for each objective (necessary)
    lb: (1 x dim)-vector of lower bound of the design space (necessary)
    ub: (1 x dim)-vector of upper bound of the design space  (necessary)
    criterion: function handle to the infill criterion (necessary).
               The function handle has to be of the form @(x)criterion(x,...)
               Please note: The additional parameters ... required by the
               criterion are those existing when the function is created
               The criterion has to return a 1x2 vector [y dy]
               y:  the value of the criterion (double to be minimized)
               dy: (1xdim)-vector of the analytical gradient at y
                   if not known, NaN should be returned
    maxEval: maximum number of evaluations (optional, default 5e4)
    tol: tolerance in detecting local optima (optional, default 1e-16)
    makeDebugOutput: boolean for analytic plots (optional, default false)
    '''
    if model1 is None or lb is None or ub is None or criterion is None:
        raise ValueError('model1, lb, ub and criterion are necessary input arguments')
    if len(model1) > 1:
        model1 = model1[0]
    N, n = model1['S'].shape
    par = model1['Ssc'][0] + model1['S'] * model1['Ssc'][1]
    #compute startPoints
    startPoints, sLB, sUB, score = computeStartPoints(model1, lb, ub, tol)

    #rank start points
    toBeSorted = np.min(score,axis=1)
    idx = np.argsort(toBeSorted)
    scoreAgg = toBeSorted[idx]
    #filter start points which perform worse than average in both scores
    scoreAggSmallerThen0 = scoreAgg<0
    startPoints = startPoints[idx[scoreAggSmallerThen0],:]
    sLB = sLB[idx[scoreAggSmallerThen0],:]
    sUB = sUB[idx[scoreAggSmallerThen0],:]
    nStartingPoints = len(startPoints)
    
    X = np.empty((nStartingPoints, n))
    X.fill(np.nan)
    Y = np.empty(nStartingPoints)
    Y.fill(np.nan)

    X2 = np.empty((nStartingPoints, n))
    X2.fill(np.nan) 
    Y2 = np.empty(nStartingPoints)
    Y2.fill(np.nan)
    
    X3 = np.empty((nStartingPoints, n))
    X3.fill(np.nan) 
    CV = np.empty(nStartingPoints)
    CV.fill(np.nan)
    start = time.time()
    
    processors = -1+multiprocessing.cpu_count()
    
    indexes = np.array(list(range(nStartingPoints)))
    pool = multiprocessing.Pool(processes=processors)
    processs = []
    for i in range(processors):
        booleani = indexes%processors==i
        startPointsI =  startPoints[booleani]
        sLBI = sLB[booleani]
        sUBI = sUB[booleani]
        processs.append((startPointsI, maxEval/processors, sLBI, sUBI, lb, ub, EPS, constraintSurrogates, criterion))
    
    results = pool.map(minimizeCriteria, processs)
    
    for i in range(len(results)):
        Xi,Yi,X2i,Y2i,X3i,CVi = results[i]
        booleani = indexes%processors==i
        X[booleani] = Xi
        Y[booleani] = Yi
        X2[booleani] = X2i
        Y2[booleani] = Y2i
        X3[booleani] = X3i
        CV[booleani] = CVi

#    evals = 0
#    i = 0    
##    partialConstraintPredictionsFunction = partial(predictConstraints2, rbfmodel=constraintSurrogates, EPS=EPS)
#    partialConstraintPredictionsFunction = partial(predictConstraints, rbfmodel=constraintSurrogates, EPS=EPS)        
#    
#    while evals < maxEval and i < nStartingPoints:
#        x0 = startPoints[i,:]
#        cons = []
#        #add constraints as function constraints
#        cons.append({'type':'ineq','fun':partialConstraintPredictionsFunction})
#        
#        #add lower and upper bound
#        for factor in range(len(sLB[i])):
#            lower = sLB[i][factor]
#            l = {'type':'ineq','fun': lambda x, lbb=lower, i=factor: x[i]-lbb}
#            cons.append(l)
#        for factor in range(len(sUB[i])):
#            upper = sUB[i][factor]
#            u = {'type':'ineq','fun': lambda x, ubb=upper, i=factor: ubb-x[i]}
#            cons.append(u)
#        
#        opts = {'maxiter':500, 'tol':1e-6, 'catol':1e-6}
#       
#        optimResult = optimize.minimize(criterion, x0, constraints=cons, options=opts, method='COBYLA')
#
#        Xtemp = optimResult.x
#        Ytemp = optimResult.fun
#        stopFlag = optimResult.success
#        maxcv = optimResult.maxcv
#        funcCount = optimResult.nfev
#        evals += funcCount
#        
#        if (stopFlag): #if cobyla succesfully terminated this is probably the best option
#            X[i,:] = np.maximum(np.minimum(Xtemp,ub),lb)
#            Y[i] = Ytemp
#        elif maxcv<=1e-6: # if xtemp has no violation
#            X2[i,:] = np.maximum(np.minimum(Xtemp,ub),lb)
#            Y2[i] = Ytemp
#        else: #else if cobyla did not found a feasible minima save x with the expected constraint violation value
#            X3[i,:] = np.maximum(np.minimum(Xtemp,ub),lb)
#            CV[i] = maxcv
#        i += 1
#        if evals >= maxEval:
#            print('max evals reached in FindAllLocalOptimaNew')
    
    print('time to compute local minima low tol',time.time()-start)
    succesfull = ~np.isnan(Y)
    X = X[succesfull]
    Y = Y[succesfull]
    index = np.argsort(Y)
    Y = Y[index]
    X = X[index]
    [X,ix] = adaptiveFiltering(X, par, Y)

    succesfull2 = ~np.isnan(Y2)
    X2 = X2[succesfull2]
    Y2 = Y2[succesfull2]
    index = np.argsort(Y2)
    Y2 = Y2[index]
    X2 = X2[index]
    [X2,ix2] = adaptiveFiltering(X2, par, Y2)

    CVfound = ~np.isnan(CV)
    X3 = X3[CVfound]
    CV = CV[CVfound]
    index3 = np.argsort(CV)
    CV = CV[index3]
    X3 = X3[index3]
    [X3,ix3] = adaptiveFiltering(X3, par, CV)
    
    
    notSeenBeforeN = -1
    if len(X)>0:
        notSeenBefore = ~np.array([x in par for x in X])
        notSeenBeforeN = sum(notSeenBefore)
    
    if notSeenBeforeN > 0:
        return(X,0)
    
    notSeenBeforeN2 = -1
    if len(X2)>0:
        notSeenBefore2 = ~np.array([x in par for x in X2])
        notSeenBeforeN2 = sum(notSeenBefore2)
        
    if notSeenBeforeN2 > 0:
        return(X2,1)

    return(X3, 2)