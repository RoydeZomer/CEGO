# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:13:14 2017

@author: r.dewinter
"""
import numpy as np
from paretoRank import paretoRank
from adaptiveFiltering import adaptiveFiltering

def computeStartPoints(model=None, lb=None, ub=None, tol=np.finfo('float').eps):
    '''
    Compute start points and corresponding box constraints by using the
    correlation function(s) defined by the model(s)
    Input:
    model current model struct holding the standardized search points S, the
          scaling parameters Ssc and the correlation function parameters theta
    lb    lower bound (box constraints) of the search space
    ub    upper bound (box constraints) of the search space
    Output:
    S     start points for the infill criterion optimization
    bcLB  lower box constraints for each start point
    bcUB  upper box constriants for each start point
    '''
    if model is None:
        raise ValueError('model is a necessary input argument')
    else:
        if type(model)==dict:
            model = [model]
    
    Xs = model[0]['S'] #S is equal for all models (vector evaluation)
    Ni,n = Xs.shape
    m = len(model)
    
    UBs = 0
    if ub is None:
        UBs = np.max(Xs, axis=0)
    else:
        #standardize given bound
        UBs = (ub-model[0]['Ssc'][0])/model[0]['Ssc'][1]
    
    LBs = 0
    if lb is None:
        LBs = min(Xs)
    else:
        #standardize given bound
        LBs = (lb-model[0]['Ssc'][0])/model[0]['Ssc'][1]
    
    Ys = 0
    if m>1:
        #use Pareto rank as aggregate objective value
        obj = np.zeros((Ni, m))
        for i in range(m):
            obj[:,i] = np.ndarray.flatten(model[i]['Y'])
        Ys = paretoRank(obj)
        Ys = (Ys-np.mean(Ys))/np.std(Ys)
    else:
        Ys = model[0]['Y']
    
    dist = np.zeros((Ni, n))
    Ns = Ni*m*(2**n)
    S = np.zeros((Ns, n))
    bcLB = np.zeros((Ns,n))
    bcUB = np.zeros((Ns,n))
    score = np.zeros((Ns,2))
    border = np.zeros(n)
    
    binary = lambda n: n>0 and [n&1]+binary(n>>1) or []
    
    for d in range(m):
        corrpar = model[d]['theta']
        if len(corrpar)==1:
            theta = corrpar*np.ones(n)
            p = 2*np.ones(n)
        elif len(corrpar)==n or len(corrpar)==n+1:
            theta= corrpar[:n]
            p = 2*np.ones(n)
        elif len(corrpar) >= 2*n:
            theta = corrpar[:n]
            p = corrpar[n:2*n]
            
        # compute start points
        for i in range(Ni-1):
            # calculate correlation distances
            for j in range(Ni-1):
                diff = Xs[j,:] - Xs[i,:]
                dist[j,:] = np.sign(diff)*theta*(abs(diff)**p)
            # detect position relative to the current decision vector
            iBin = (dist>0)
            iInt = np.sum((iBin*np.power(2,(list(range(0,n))))), axis=1)
            # assign start points and constraints in each possible direction
            for j in range(2**n):
                ix = (i)*(2**n)+j+1 #current index
                bina = binary(j)
                bina = bina+(n-len(bina))*[0]
                curSign = -1*np.ones(n)+2*np.array(bina)
                # find solutions in the current relative position, zero distances
                # are always considered
                active = ((iInt==j) | np.sum(dist==0, axis=1)>0) & (np.sum(dist==0, axis=1)!=len(dist[0]))
                for k in range(n):
                    if curSign[k] > 0:
                        border[k] = UBs[k]
                    else:
                        border[k] = LBs[k]
                minDist = abs(border-Xs[i,:])
                if any(active):
                    arr = np.max(abs(dist[active,:]), axis=1)
                    minDistNorm = np.min(arr)
                    minIx = np.argwhere(arr==minDistNorm)[0]
                    #transformation to untransformed coordinate distance
                    minDistTry = (minDistNorm/theta)**(1/p)
                    minDist = np.array([min(minDistTry[a],minDist[a]) for a in range(len(minDistTry))])
                    #start point lies halfway between point and corner
                    S[ix,:] = Xs[i,:] + 0.5*curSign*minDist
                else:
                    #start point lies in the corner
                    S[ix,:] = Xs[i,:] + curSign*minDist
                
                for k in range(n):
                    if curSign[k] > 0:
                        bcLB[ix,k] = Xs[i,k]
                        bcUB[ix,k] = Xs[i,k] + minDist[k]
                    else:
                        bcLB[ix,k] = Xs[i,k] - minDist[k]
                        bcUB[ix,k] = Xs[i,k]
                if any(active):
                    tempY = Ys[active]
                    score[ix, :] = [0.67*Ys[i]+0.33*tempY[minIx], sum(0.5*minDist)]
                else:
                    score[ix, :] = [Ys[i], sum(minDist)]
    
    [S, ix] = adaptiveFiltering(S, Xs, score[:,0])
    bcLB = bcLB[ix]
    bcUB = bcUB[ix]
    score = score[ix]
    Ns = sum(ix)
    S = model[0]['Ssc'][0,:] + S*model[0]['Ssc'][1,:]
    bcLB = model[0]['Ssc'][0,:] + bcLB*model[0]['Ssc'][1,:]
    bcUB = model[0]['Ssc'][0,:] + bcUB*model[0]['Ssc'][1,:]
    active = np.all(bcUB-bcLB > 2*np.sqrt(tol), axis=1)
    S = S[active]
    
    bcLB = bcLB[active]
    bcUB = bcUB[active]
    score = score[active]
    score[:,1] = -(score[:,1] - np.mean(score[:,1]))/np.std(score[:,1])
    
    return [S, bcLB, bcUB, score]
