# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:08:46 2017

@author: r.dewinter
"""
from Likelihood import Likelihood
from functools import partial
from fminsearchbnd import fminsearchbnd
from regpoly0 import regpoly0
from corrnoisykriging import corrnoisykriging
from dacefitEGO import dacefitEGO

import numpy as np

def simplexKriging(parameters=None, objectives=None, startTheta=None, tol=None):
    '''
    D. Gaida: changed. added outdir and further as arguments
    function [model like nEval] = ...
        simplexKriging(parameters, objectives, startTheta, tol, outdir, i, ...
        load_from_file, save_to_file)
    Compute a DACE model for given input and output using correlation
    parameters optimized by a direct simplex optimization with smart
    starting parameters (recommended for modelling deterministic or
    only slightly noisy data)
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
    
    
    D. Gaida added
    throw Infs/NaNs out (those are failed simulations)
    arbeite hier mit NaNs da simBiogasPlant NaNs schreibt bei
    fehlgeschlagenen Simulationen. der grund ist, dass sonst in SMSEGO der
    ref Punkt Inf wird und damit kein feasible punkt ist für das nachfolgende
    optimierungsproblem
    '''
    n,m = parameters.shape
    lb = np.ones(m)*-12
    lb = np.append(lb, np.ones(m)*0.01)
    ub = np.ones(m)*10
    ub = np.append(ub, np.ones(m)*2)
    nugget = -1
    if parameters is None:
        raise ValueError('parameters and objectives are necessary input arguments')
    if objectives is None:
        raise ValueError('parameters and objectives are necessary input arguments')
    if startTheta is None:
        startTheta = np.ones(m)*(n/(100*m))
        startTheta = np.append(startTheta, np.ones(m)*1.9)
        startTheta = np.append(startTheta,0.999)
        tol = 1e-6
    else:
        if len(startTheta) < 2*m+1:
            if len(startTheta) == 1:
                nugget = startTheta[0]
                startTheta = np.ones(m)*(n/(100*m))
                startTheta = np.append(startTheta, np.ones(m)*1.9)
                startTheta = np.append(startTheta,0.999)
            elif len(startTheta) == 2*m:
                startTheta = np.append(startTheta, 0.999)
            else:
                raise ValueError('startTheta has to be a vector of length 1, 2*n, or 2*n+1')
        elif startTheta[2*m] == 1: #no noise
            nugget = 1
        if tol is None:
            tol = 1e-6
    
    if nugget > 0 and nugget <= 1:
        startTheta1 = np.log10(startTheta[:m])
        startTheta = np.append(startTheta1, startTheta[m:2*m])
        q = len(startTheta)
        opt = {'maxiter':5000*q, 'fatol':tol, 'maxfev':1000*q}
        [bestTheta, like, flag, out] = fminsearchbnd(partial(Likelihood, parameters=parameters, objectives=objectives, nugget=nugget), startTheta, lb, ub, opt)
        thetaConv = np.append(np.power(10, bestTheta[:m]), bestTheta[m:2*m])
        thetaConv = np.append(thetaConv, nugget)
    else:
        startTheta1 = np.array(np.log10(startTheta[:m]))
        startTheta = np.append(startTheta1, startTheta[m:2*m+1])
        lb = np.append(lb,0.5)
        ub = np.append(ub,1)
        q = len(startTheta)
        opt = {'maxiter':5000*q, 'fatol':tol, 'maxfev':1000*q}
        [bestTheta, like, flag, out] = fminsearchbnd(partial(Likelihood, parameters=parameters, objectives=objectives), startTheta, lb, ub, opt)
        thetaConv = np.append(np.power(10, bestTheta[:m]), bestTheta[m:2*m+1])
        
    model = dacefitEGO(parameters, objectives, regpoly0, corrnoisykriging, thetaConv)[0]
    nEval = out['funcount']
    
    
    return [model, like, nEval]