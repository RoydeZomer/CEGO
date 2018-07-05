# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 14:40:01 2017

@author: r.dewinter
"""
from Likelihood import Likelihood
from functools import partial
from fminsearchbnd import fminsearchbnd
from regpoly0 import regpoly0
from corrnoisygauss import corrnoisygauss
from dacefitEGO import dacefitEGO

import numpy as np

def simplexGauss(parameters=None, objectives=None, startTheta=None, tol=None):
    """
    Compute a DACE model for given input and output using correlation
    parameters optimized by a direct simplex optimization with smart
    starting parameters
    Call: [model like] = dacefitNew(parameters, objectives, startTheta)
    Output parameters:
    model: DACE model struct with optimized correlation parameters
    like: Probability of the data under the model (optional)
    Input parameters:
    parameters: input matrix (n x dim) (necessary)
    objectives: output vector (n x 1) to the given input (necessary)
    startTheta: (scalar) determine self correlation (interpolation)
                (vector) initial guess of the model parameters
    tol: tolerance in detecting the optimal model parameters (optional)
    """
    n,m = parameters.shape
    lb = np.ones(m)*-12
    ub = np.ones(m)*10
    nugget = -1
    if parameters is None:
        raise ValueError('parameters and objectives are necessary input arguments')
    if objectives is None:
        raise ValueError('parameters and objectives are necessary input arguments')
    if startTheta is None:
        startTheta = np.ones(m)*(n/(100*m))
        startTheta = np.append(startTheta,0.999)
        tol = 1e-6
    else:
        if len(startTheta) < m+1:
            if len(startTheta) == 1:
                nugget = startTheta[0]
                startTheta = np.ones(m)*(n/(100*m))
                startTheta = np.append(startTheta,0.999)
            elif len(startTheta) == m:
                startTheta = np.append(startTheta, 0.999)
            else:
                raise ValueError('startTheta has to be a vector of length 1, n, or n+1')
        elif startTheta[m] == 1:
            nugget = 1
        if tol is None:
            tol = 1e-6
    
    if nugget > 0 and nugget <= 1:
        startTheta = np.log10(startTheta[:m])
        q = len(startTheta)
        opt = {'maxiter':5000*q, 'fatol':tol, 'maxfev':1000*q}
        [bestTheta, like, flag, out] = fminsearchbnd(partial(Likelihood, parameters=parameters, objectives=objectives, nugget=nugget), startTheta, lb, ub, opt)
        thetaConv = np.append(np.power(10, bestTheta[:m]), nugget)
    else:
        last = startTheta[m]
        startTheta = np.array(np.log10(startTheta[:m]))
        startTheta = np.append(startTheta, last)
        lb = np.append(lb,0.5)
        ub = np.append(ub,1)
        q = len(startTheta)
        opt = {'maxiter':5000*q, 'fatol':tol, 'maxfev':1000*q}
        [bestTheta, like, flag, out] = fminsearchbnd(partial(Likelihood, parameters=parameters, objectives=objectives), startTheta, lb, ub, opt)
        thetaConv = np.append(np.power(10, bestTheta[:m]), bestTheta[m])
        
    model = dacefitEGO(parameters, objectives, regpoly0, corrnoisygauss, thetaConv)[0]
    nEval = out['funcount']
    if nEval==1000*q:
        print('maxfev reached simplexgauss')
    return [model, like, nEval]
