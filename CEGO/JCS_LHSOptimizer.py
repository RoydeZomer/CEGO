# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 10:57:13 2017

@author: r.dewinter
"""
import numpy as np
from evaluateEntropy import evaluateEntropy
from lhs import lhs



def JCS_LHSOptimizer(n, m, no_eval=None, theta=None, levels=None):
    '''
    JCS_LHSOptimizer(n, m, doPlot, no_eval, theta)
    ----------------------------------------------------------------------
    Find LHS optimized respective to the Entropy Criterion using the
    algorithm of Jin, Chen and Sudjijanta (Published in
    Statistical Planning and Inference 134 (2005) p. 268 - 287)
    
    Output arguments
    bestLHS: sizeLHS x n design matrix, which is optimal respective to entropy
    fBest: entropy of final LHS (optional)
    nEval: number of evaluations needed (optional)
    Input arguments
    n: number of designs in the output LHS (necessary)
    m: number of dimensions per design (necessary)
    no_eval: maximum number of trials for determining the LHS (optional)
    theta: weights and exponents of the correlation function (optional)
    levels: number of factor levels (optional, default n)
    '''
    if theta is None:
        d = 1 / (n**(1/m)-1)
        eps = np.finfo(float).eps
        theta = np.array([-np.log(eps)/(m*np.power(d,2))]*m)
        if no_eval is None:
            no_eval = min(10^(m+3), 1e6)
    elif (len(theta) != m) and (len(theta) != 2*m) and (len(theta) != 2*m+1):
        raise ValueError('theta has to be of length m, 2m or 2m+1')
    
    LHS = lhs(m, samples=n, criterion="center", iterations=5)

    if levels is not None:
        if len(levels) == 1:
            levels = m*list(levels)
        elif len(levels) != m:
            raise ValueError('levels has to be of length m or 1')
        levelsMMatrix = np.array([list(levels)]*n)
        levelsDMatrix = np.array([list(levels -1)]*n)
        LHS = ((LHS*levelsMMatrix).astype(int))/levelsDMatrix
    
    fBest = evaluateEntropy(LHS, theta)
    bestLHS = LHS
    fCur = fBest
    curLHS = bestLHS
    Th = 0.005 * fBest #initial threshold
    allperm = [[x,x2] for x in range(n) for x2 in range(x,n) if x != x2]
    allperm = np.array(allperm)
    ne = len(allperm)
    J = min(int(ne/5), 50)
    M = int(np.ceil(min(2*n/J, 100/m) )*m)
    max_iterations = np.ceil(no_eval / (M*J))
    alpha1 = 0.8 # set according to Jin et al.
    alpha2 = 0.9 # set according to Jin et al.
    alpha3 = 0.7 # set according to Jin et al.
    up = False
    down = False
    noImprCycles = 0
    j = 0
    while (noImprCycles <= min(3,m+1)) and (j <= max_iterations):
        #outer step1 update
        fOldBest = fBest
        nAcpt = 0
        nImp = 0
        #outer step2 start inner loop
        for i in range(M):
            #inner step 1 random pick j distinct element exchanges
            index = np.random.permutation(ne)
            ii = ((i-1)%m)
            #inner step2 choose the best element exchange
            LHS = curLHS
            temp = LHS[ allperm[index[0],0] , ii]
            LHS[allperm[index[0],0],ii] = LHS[allperm[index[0],1],ii]
            LHS[allperm[index[0],1],ii] = temp
            fTry = evaluateEntropy(LHS, theta)
            tryLHS = LHS
            for x in range(1,J):
                LHS = curLHS
                temp = LHS[allperm[index[x],0],ii]
                LHS[allperm[index[x],0],ii] = LHS[allperm[index[x],1],ii]
                LHS[allperm[index[x],1],ii] = temp
                #calculateCorrMatrix
                temp = evaluateEntropy(LHS, theta)
                if temp < fTry:
                    fTry = temp
                    tryLHS = LHS
            #inner step3 compare to curLHS and bestLHS
            if fTry - fCur < Th*np.random.rand():
                curLHS = tryLHS
                fCur = fTry
                nAcpt = nAcpt + 1
                if fTry < fBest:
                    bestLHS = tryLHS
                    fBest = fTry
                    nImp = nImp + 1
        #outer step3 determine current kind of process and update Th
        acptRatio = nAcpt / M
        if fBest < fOldBest:
            noImprCycles = 0
            if acptRatio > 0.1:
                impRatio = nImp / M
                if impRatio < acptRatio:
                    Th = Th * alpha1 # decrease Th
            else:
                Th = Th / alpha1 # increase Th
        else:
            # exploration process
            if up or (acptRatio < 0.1):
                Th = Th / alpha3 # rapidly increase Th
                if not up:
                    noImprCycles = noImprCycles + 1
                    up = True
                    down = False
            if down or (acptRatio > 0.8):
                Th = Th * alpha2 # slowly decrease Th
                if not down:
                    down = True
                    up = False
        j += 1
        
    j = j * M * J
    return[bestLHS, fBest, j]