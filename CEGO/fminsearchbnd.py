# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 09:33:14 2017

@author: r.dewinter
"""
import numpy as np
from scipy.optimize import minimize

def fminsearchbnd(fun=None,x0=None,LB=None,UB=None,options=None,varargin=None):
    '''
    FMINSEARCHBND: FMINSEARCH, but with bound constraints by transformation
    usage: x=FMINSEARCHBND(fun,x0)
    usage: x=FMINSEARCHBND(fun,x0,LB)
    usage: x=FMINSEARCHBND(fun,x0,LB,UB)
    usage: x=FMINSEARCHBND(fun,x0,LB,UB,options)
    usage: x=FMINSEARCHBND(fun,x0,LB,UB,options,p1,p2,...)
    usage: [x,fval,exitflag,output]=FMINSEARCHBND(fun,x0,...)
    
    arguments:
     fun, x0, options - see the help for FMINSEARCH
    
     LB - lower bound vector or array, must be the same size as x0
    
          If no lower bounds exist for one of the variables, then
          supply -inf for that variable.
    
          If no lower bounds at all, then LB may be left empty.
    
          Variables may be fixed in value by setting the corresponding
          lower and upper bounds to exactly the same value.
    
     UB - upper bound vector or array, must be the same size as x0
    
          If no upper bounds exist for one of the variables, then
          supply +inf for that variable.
    
          If no upper bounds at all, then UB may be left empty.
    
          Variables may be fixed in value by setting the corresponding
          lower and upper bounds to exactly the same value.
    
    Notes:
    
     If options is supplied, then TolX will apply to the transformed
     variables. All other FMINSEARCH parameters should be unaffected.
    
     Variables which are constrained by both a lower and an upper
     bound will use a sin transformation. Those constrained by
     only a lower or an upper bound will use a quadratic
     transformation, and unconstrained variables will be left alone.
    
     Variables may be fixed by setting their respective bounds equal.
     In this case, the problem will be reduced in size for FMINSEARCH.
    
     The bounds are inclusive inequalities, which admit the
     boundary values themselves, but will not permit ANY function
     evaluations outside the bounds. These constraints are strictly
     followed.
    
     If your problem has an EXCLUSIVE (strict) constraint which will
     not admit evaluation at the bound itself, then you must provide
     a slightly offset bound. An example of this is a function which
     contains the log of one of its parameters. If you constrain the
     variable to have a lower bound of zero, then FMINSEARCHBND may
     try to evaluate the function exactly at zero.
    
    
    Example usage:
    rosen = @(x) (1-x(1)).^2 + 105*(x(2)-x(1).^2).^2;
    
    fminsearch(rosen,[3 3])     unconstrained
    ans =
       1.0000    1.0000
    
    fminsearchbnd(rosen,[3 3],[2 2],[])     constrained
    ans =
       2.0000    4.0000
    
    See test_main.m for other examples of use.
    
    
    See also: fminsearch, fminspleas
    
    
    size checks
    '''
    n = len(x0)
    
    if LB is None:
        LB = np.full_like(np.empty(n), -np.inf)
    if UB is None:
        UB = np.full_like(np.empty(n), np.inf)
    
    if n!=len(LB) or n!=len(UB):
        raise ValueError('x0 is incompatible in size with either LB or UB')
    
    if options is None or not options:
        options = dict()
        options['Display'] = True
        options['maxiter'] = 200*n
        options['fatol'] = 1e-4
    
    params = dict()
    params['args'] = varargin
    params['LB'] = LB
    params['UB'] = UB
    params['fun'] = fun
    params['n'] = n
    params['OutputFcn'] = []
    
    # 0 --> unconstrained variable
    # 1 --> lower bound only
    # 2 --> upper bound only
    # 3 --> dual finite bounds
    # 4 --> fixed variable
    boundClass = np.zeros(n)
    for i in range(n):
        k = np.isfinite(LB[i]) + 2*np.isfinite(UB[i])
        boundClass[i] = k
        if k==3 and LB[i] == UB[i]:
            boundClass[i] = 4
    
    params['BoundClass'] = boundClass
    
    x0u = np.copy(x0)
    k = 0
    for i in range(n):
        bC = params['BoundClass'][i]
        if bC == 1:
            if x0[i] <= LB[i]:
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(x0[i] - LB[i])
            k+=1
        if bC == 2:
            if x0[i]>=UB[i]:
                x0u[k] = 0
            else:
                x0u[k] = np.sqrt(UB[i]-x0[i])
            k+=1
        if bC == 3:
            if x0[i]<=LB[i]:
                x0u[k] = -np.pi/2
            elif x0[i]>=UB[i]:
                x0u[k] = -np.pi/2
            else:
                x0u[k] = 2*(x0[i] - LB[i])/(UB[i] - LB[i]) -1
                x0u[k] = 2*np.pi+np.arcsin(max(-1,min(1,x0u[k])))
            k+=1
        if bC == 0:
            x0u[k] = x0[i]
            k+=1
        #dont do anything if bC == 4
    if k<n:
        x0u = x0u[:k]
    
    if len(x0u)==0:
        x = xtransform(x0u,params)
        fval = params['fun'](x)
        exitflag = False
        output = dict()
        output['iterations'] = 0
        output['funcount'] = 1
        output['algorithm'] = 'fminsearch'
        output['message'] = 'All variables were held fixed by the applied bounds'
        return [x, fval, exitflag, output]

    
    def outfun_wrapper(x, varargin, params):
        xtrans = xtransform(x,params)
        stop = params['OutputFcn'](xtrans,varargin)
        return stop
    
    if 'OutputFcn' in options:
        params['OutputFcn'] = options['OutputFcn']
        options['OutputFcn'] = outfun_wrapper
        
    optimizeResult = minimize(intrafun,x0u,args=(params),method='Nelder-Mead',tol=np.inf, options=options)
    fval = optimizeResult['fun']
    exitflag = optimizeResult['success']
    xu = optimizeResult['x']
    output = dict()
    output['iterations'] = optimizeResult['nit']
    output['funcount'] = optimizeResult['nfev']
    output['algorithm'] = 'fminsearch'
    output['message'] = optimizeResult['message']
    
    x = xtransform(xu,params)
    
    return [x, fval, exitflag, output]

def intrafun(x, params=None):
    xtrans = xtransform(x,params)
    fval = params['fun'](xtrans)
    return fval


def xtransform(x,params):
    xtrans = np.zeros(params['n'])
    k = 0
    for i in range(params['n']):
        bC = params['BoundClass'][i]
        if bC == 1:
            xtrans[i] = params['LB'][i] + x[k]**2
            k += 1
        if bC == 2:
            xtrans[i] = params['UB'][i] - x[k]**2
            k += 1
        if bC == 3:
            xtrans[i] = (np.sin(x[k])+1)/2
            xtrans[i] = xtrans[i]*(params['UB'][i] - params['LB'][i]) + params['LB'][i]
            xtrans[i] = max(params['LB'][i], min(params['UB'][i],xtrans[i]))
            k+=1
        if bC == 4:
            xtrans[i] = params['LB'][i]
        if bC == 0:
            xtrans[i] = x[k]
            k+=1
    return xtrans