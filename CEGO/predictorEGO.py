# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 15:12:38 2017

@author: r.dewinter
"""
import numpy as np

def predictorEGO(x=None, dmodel=None):
    #PREDICTOR  Predictor for y(x) using the given DACE model.
    #
    # Call:   y = predictor(x, dmodel)
    #         [y, or] = predictor(x, dmodel)
    #         [y, dy, mse] = predictor(x, dmodel)
    #         [y, dy, mse, dmse] = predictor(x, dmodel)
    #
    # Input
    # x      : trial design sites with n dimensions.
    #          For mx trial sites x:
    #          If mx = 1, then both a row and a column vector is accepted,
    #          otherwise, x must be an mx*n matrix with the sites stored
    #          rowwise.
    # dmodel : Struct with DACE model; see DACEFIT
    #
    # Output
    # y    : predicted response at x.
    # or   : If mx = 1, then or = gradient vector/Jacobian matrix of predictor
    #        otherwise, or is an vector with mx rows containing the estimated
    #                   mean squared error of the predictor
    # Three or four results are allowed only when mx = 1,
    # dy   : Gradient of predictor; column vector with  n elements
    # mse  : Estimated mean squared error of the predictor;
    # dmse : Gradient vector/Jacobian matrix of mse
    
    # Original code by Hans-Brunn Nielsen: hbn@imm.dtu.dk
    #
    # additional code for generalization to different models by Tobias Wagner:
    # wagner@isf.de
    or1 = None
    or2 = None
    sx = x.shape
    if sx[0] > 5000:
        y = np.zeros(sx[0])
        or1 = np.zeros(sx[0])
        raise ValueError('sx > 5000 dit moet je nog maken!')
        #do something
        return [y, or1, or2]
    
    if 'beta' not in dmodel:
        y = None
        raise ValueError('DMODEL has not been found')
    
    m,n = dmodel['S'].shape
    if len(sx) == 1 and n>1:
        nx = max(sx)
        if nx == n:
            mx = 1
    else:
        mx = sx[0]
        nx = sx[1]
    if nx != n:
        raise ValueError("Dimension of trial sites should be", n)
    
    x = (x - dmodel['Ssc'][0]) / np.tile(dmodel['Ssc'][1],(mx,1) ) 
    
    if dmodel['Ysc'].ndim>1:
        q = dmodel['Ysc'].shape[1]
    else:
        q = 1
    y = np.zeros((mx,q))
    
    if mx == 1: #one site only
        dx = x - dmodel['S']
        [f, df] = dmodel['regr'](x)
        [r, dr] = dmodel['corr'](dmodel['theta'], dx)
        dy = (df*dmodel['beta']) + np.matmul(dmodel['gamma'].T,dr)
        or1 = dy*dmodel['Ysc'][1]/dmodel['Ssc'][1,:]
        if q == 1:
            or1 = or1.T

        rt = np.linalg.solve(dmodel['C'],r)
        u = np.matmul(dmodel['Ft'].T,rt)-f.T
        v = np.linalg.solve([[dmodel['G']]],u)
        or2 = np.dot(dmodel['sigma2'], (1+np.sum(v**2)-sum(rt**2)).T)
                    
        sy = f*dmodel['beta'] + np.dot(dmodel['gamma'].T, r).T
        sy = sy[0][0]
        y = (dmodel['Ysc'][0] + dmodel['Ysc'][1]*sy).T
    else:
        dx = np.zeros((mx*m,n))
        kk = np.array(range(m))
        for k in range(mx):
            dx[kk,:] = x[k,:] - dmodel['S']
            kk = kk+m
        f = dmodel['regr'](x)[0]
        r = dmodel['corr'](dmodel['theta'], dx)[0]
        r = np.reshape(r,(m,mx))
        
        sy = (np.dot(f,dmodel['beta']).T+np.dot(dmodel['gamma'],r)).T
        y = dmodel['Ysc'][0] + dmodel['Ysc'][1]*sy
        
        rt = np.linalg.solve(dmodel['C'],r)
        u = np.linalg.solve([[dmodel['G']]], np.dot(dmodel['Ft'].T,rt)-f.T)
        
        or1 = dmodel['sigma2'] * (1+np.sum(np.power(u,2),axis=0)-np.sum(np.power(rt,2),axis=0))
        
    return [y, or1, or2]