# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:27:38 2017

@author: r.dewinter
"""
import numpy as np

def dacefitEGO(S=None,Y=None,regr=None,corr=None,theta0=None,lob=None,upb=None):
    '''
    DACEFIT Constrained non-linear least-squares fit of a given correlation
    model to the provided data set and regression model
    
    Call
       [dmodel, perf] = dacefit(S, Y, regr, corr, theta0)
       [dmodel, perf] = dacefit(S, Y, regr, corr, theta0, lob, upb)
    
    Input
    S, Y    : Data points (S(i,:), Y(i,:)), i = 1,...,m
    regr    : Function handle to a regression model
    corr    : Function handle to a correlation function
    theta0  : Initial guess on theta, the correlation function parameters
    lob,upb : If present, then lower and upper bounds on theta
               Otherwise, theta0 is used for theta
    
    Output
    dmodel  : DACE model: a struct with the elements
        regr   : function handle to the regression model
        corr   : function handle to the correlation function
        theta  : correlation function parameters
        beta   : generalized least squares estimate
        gamma  : correlation factors
        sigma2 : maximum likelihood estimate of the process variance
        S      : scaled design sites
        Ssc    : scaling factors for design arguments
        Ysc    : scaling factors for design ordinates
        C      : Cholesky factor of correlation matrix
        Ft     : Decorrelated regression matrix
        G      : From QR factorization: Ft = Q*G' .
    perf    : struct with performance information. Elements
        nv     : Number of evaluations of objective function
        perf   : (q+2)*nv array, where q is the number of elements 
                 in theta, and the columns hold current values of
                     [theta;  psi(theta);  type]
                 |type| = 1, 2 or 3, indicate 'start', 'explore' or 'move'
                 A negative value for type indicates an uphill step
    '''
    m,n = S.shape
    lY = Y.shape[0]
    if m != lY:
        raise ValueError('S and Y must have the same number of rows') 
    
    #optimisation case
    lth = len(theta0)
    if lob is not None and upb is not None:
        if len(lob) != lth or len(upb) != lth:
            raise ValueError('theta0, lob and upb must have the same length')
        if np.any(lob<=0) or np.any(upb < lob):
            raise ValueError('The bounds must satisfy  0 < lob <= upb')
    elif np.any(theta0 <= 0):
        raise ValueError('theta0 must be strictly positive')
    
    mS = np.mean(S,axis=0)
    mY = np.mean(Y,axis=0)
    sS = np.std(S,axis=0)
    sY = np.std(Y,axis=0)
    
    j = np.where(sS==0)[0]
    if len(j)!=0:
        sS[j] = 1
    j = np.where(sY==0)[0]
    if len(j)!=0:
        sY[j] = 1
    
    S = (S - mS) / sS
    Y = (Y - mY) / sY
        
    mzmax = int(m*(m-1)/2)
    ij = np.array([[x,y] for x in range(m) for y in range(x+1,m)])
    D = np.zeros((mzmax,n))
    ll = [-1]
    
    for k in range(m-1):
        start = ll[len(ll)-1]+1
        end = ll[len(ll)-1] + m-k
        ll = list(range(start,end))
        sk = np.tile(S[k,:], ((m-1)-k,1) )
        s = S[k+1:m,:]
        D[ll,:] = sk - s
    
    if not (corr.__name__=='corrnoisygauss' or corr.__name__=='corrnoisykriging') and min(sum(abs(D.T)))==0:
        raise ValueError('Multiple design sites are not allowed')
        
    [F, df] = regr(S)
    mF, p = F.shape
    if mF != m:
        raise ValueError('number of rows in  F  and  S  do not match')
    if p > mF:
        raise ValueError('least squares problem is underdetermined')
    
    par = dict()
    par['corr'] = corr
    par['regr'] = regr
    par['y'] = Y
    par['F'] = F
    par['D'] = D
    par['ij'] = ij
    par['scS'] = sS
    

    [f, fit] = objfunc(theta0,par)
    perf = dict()
    perf['perf'] = np.append(np.append(theta0,f),1)
    perf['nv'] = 1
    if f == np.inf:
        raise ValueError('Bad point. Try increasing theta0')
    
    dmodel = dict()
    dmodel['regr'] = regr
    dmodel['corr'] = corr
    dmodel['theta'] = theta0
    dmodel['beta'] = fit['beta']
    dmodel['gamma'] = fit['gamma']
    dmodel['sigma2'] = np.power(sY,2)*fit['sigma2']
    dmodel['S'] = S
    dmodel['Ssc'] = np.array([mS, sS])
    dmodel['Y'] = Y
    dmodel['Ysc'] = np.array([mY, sY])
    dmodel['C'] = fit['C']
    dmodel['Ft'] = fit['Ft']
    dmodel['G'] = fit['G']
    dmodel['detR'] = fit['detR']
    
    return [dmodel, perf]

def objfunc(theta=None, par=None):
    obj = np.inf
    fit = dict()
    fit['sigma2'] = None
    fit['beta'] = None
    fit['gamma']= None
    fit['C'] = None
    fit['Ft'] = None
    fit['G'] = None
    m = par['F'].shape[0]
    r = par['corr'](theta, par['D'])[0]
    mask = r > 0
    
    o = np.array(range(m))
    mu = (10+m)*np.finfo('float').eps
    
    arg1 = np.append(par['ij'][:,0][mask], o)
    arg2 = np.append(par['ij'][:,1][mask], o)
    value = np.append(r[mask], np.ones(m)+mu)
    
    dim = max(max(arg1),max(arg2))+1
    R = np.zeros((dim,dim))
    R[arg1,arg2] = value
    
    
    if not np.all(np.linalg.eigvals(R.T)):
        ValueError('R.T is not positive definite')
        return [obj, fit]
    
    C = cholesky_d(R.T)
    
    Ft = np.linalg.solve(C, par['F'])
    
    Q , G = np.linalg.qr(Ft)
    
    if np.linalg.cond(G) < 1e-10:
        if np.linalg.cond(par['F'])>1e15:
            raise ValueError('F is too ill conditioned\nPoor combination of regression model and design sites')
        else:
            return [obj, fit]
    
    Yt = np.linalg.solve(C, par['y'])
    beta = np.linalg.solve(G, (np.dot(Q.T,Yt)))
    rho = Yt - np.dot(Ft,beta)
    sigma2 = np.sum(np.power(rho,2))/m
    detR = np.prod( np.power(np.diag(C), (2/m) ) )
    obj = np.sum(sigma2)*detR
    
    fit = dict()
    fit['sigma2'] = sigma2
    fit['beta'] = beta[0]
    try:
        #    fit['gamma'] = np.dot(np.ndarray.flatten(rho),scipy.linalg.pinv(C))
        fit['gamma'] = np.dot(np.ndarray.flatten(rho),np.linalg.pinv(C))
    except:
        fit['gamma'] = None
            
    fit['C'] = C
    fit['Ft'] = Ft
    fit['G'] = G[0][0]
    fit['detR'] = detR
    
    return [obj, fit]
    
def cholesky_d(A):
    L = np.zeros_like(A)
    n = len(L)
    for i in range(n):
        for j in range(i+1):
            if i==j:
                val = A[i,i] - np.sum(np.square(L[i,:i]))
                # if diagonal values are negative return zero - not throw exception
                if val<0:
                    return 0.0
                L[i,i] = np.sqrt(val)
            else:
                L[i,j] = (A[i,j] - np.sum(L[i,:j]*L[j,:j]))/L[j,j]
                
    return L