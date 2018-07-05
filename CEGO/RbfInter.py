# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:58:58 2018

@author: r.dewinter
"""

import numpy as np
from scipy.spatial.distance import pdist
import time

import warnings

def adjustDRC(objective):
    FR = np.max(objective) - np.min(objective)
    if FR > 1000:
        return np.array([0.001,0.0])
    else:
        return np.array([0.3,0.05,0.001,0.0005,0.0])

def trainCubicRBF(xp, U, lb, ub, objective, squares=True, rho=0.0):
    '''
    Fit cubic RBF interpolation to training data for d>1.
    
    The model for a point z=(z_1,...,z_d) is fitted using n sample points x_1, ..., x_n 
       s(z) = \lambda_1*\Phi(||z-x_1||)+... +\lambda_n*\Phi(||z-x_n||)
                     + c_0 + c_1*z_1 + ... + c_d*z_d  
    
    where \Phi(r)=r^3 denotes the cubic radial basis function. The coefficients \lambda_1, 
    ..., \lambda_n, c_0, c_1, ..., c_d are determined by this training procedure.
    This is for the default case squares==FALSE. In case squares==TRUE 
    there are d additional pure square terms and the model is
    
       s_sq(z) = s(z) + c_d+1*z_1^2 + ... + c_d+d*z_d^2  
    
      
    The linear equation system is solved via SVD inversion. Near-zero elements 
    in the diagonal matrix D are set to zero in D^-1. This is numerically stable 
    for rank-deficient systems.
    
    @param xp      n points x_i of dimension d are arranged in (n x d) matrix xp
    @param U       vector of length n, containing samples u(x_i) of 
                   the scalar function u to be fitted 
                   - or - 
                   (n x m) matrix, where each column 1,...,m contains one vector of samples
                   u_j(x_i) for the m'th model, j=1,...,m
    @param squares [True] flag, see description
    @param rho     [0.0] experimental: 0: interpolating, >0, approximating (spline-like) 
                   Gaussian RBFs
                   
    @return rbf.model,  an object of class RBFinter, which is basically a list 
    with elements:
         coef  (n+d+1 x m) matrix holding in column m the coefficients for the m'th 
                       model:      \lambda_1, ..., \lambda_n, c_0, c_1, ..., c_d.  
                       In case squares==TRUE it is an (n+2d+1 x m) matrix holding  
                       additionally the coefficients c_d+1, ..., c_d+d.                    
         xp  matrix xp   
         d  dimension d 
         npts  number n of points x_i 
         squares  TRUE or FALSE  
         type  "CUBIC"
    '''
    if xp is None or U is None or lb is None or ub is None:
        raise ValueError('trainCubicRBF requires at least four arguments (xp, U, rngMin, rngMax)')
    
    xp = (2)*((xp-lb)/(ub-lb))-1

    def svdInv(M):
        eps = 1e-14
        u,s,v = np.linalg.svd(M)
        invD = 1/s
        invD[abs(s/s[0])<eps] = 0
        invM = np.matmul(v.T,np.matmul(np.diag(invD),u.T))
        return(invM)
    
    d = len(xp[0])
    if squares:
        d = d+d
    npts = len(xp)
    distances = pdist(xp)
    edits = np.zeros((npts,npts)) #euclidean distance matrix
    l = 0
    for i in range(npts):
        for j in range(i+1,npts):
            edits[i,j] = distances[l]
            edits[j,i] = distances[l]
            l+=1
    phi = np.multiply(np.multiply(edits,edits),edits) # cubic RBF (npts x npts matrix)
                                                      # /WK/ experimental: rho>0 means spline approximation
    diag = np.zeros((npts,npts))
    np.fill_diagonal(diag,1)
    phi = phi - diag*npts*rho # /WK/ instead of exact interpolating RBF (rho=0)
    pMat = np.column_stack((np.ones((npts,1)),xp)) # linear tail LH(1,x1,x2,...)
    if squares:
        pMat = np.column_stack((pMat,np.multiply(xp,xp))) # ... plus direct squares x1^2, x2^2, ...
    nMat = np.zeros((d+1,d+1))
    
    m1 = np.column_stack((phi,pMat))
    m2 = np.column_stack((pMat.T,nMat))
    M = np.row_stack((m1,m2))
    if U.ndim == 1:
        rhs = np.append(U,np.zeros(d+1))
    elif U.ndim == 2:
        rhs = np.row_stack((U,np.zeros((d+1,len(U[0])))))
    else:
        raise ValueError('U is neither vector nor matirx!')
    
    invM = svdInv(M)
    coef = np.matmul(invM,rhs)
    
    rbfmodel = dict()
    rbfmodel['coef'] = coef
    rbfmodel['xp'] = xp
    rbfmodel['d'] = d
    rbfmodel['XI'] = adjustDRC(objective)
    rbfmodel['lb'] = lb
    rbfmodel['ub'] = ub
    
    return rbfmodel

def distLine(x, xp):
    z = np.outer(np.ones(len(xp)), x) - xp
    z = np.sqrt(np.sum(z*z,axis=1))
    return z

def interpolateRBF(x, rbfModel, squares=True):
    '''
    Apply cubic or Gaussian RBF interpolation to new data for d>1.
    
    param x         vector holding a point of dimension d
    param rbf.model trained RBF model (or set of models), see trainCubicRBF
                     or trainGaussRBF
     @param squares [True] flag, see description
                   
    return          value s(x) of the trained model at x
                     - or - 
                     vector s_j(x) with values for all trained models j=1,...,m at x
    
    seealso   trainCubicRBF, predict.RBFinter
    '''
    if x.shape[1]!=len(rbfModel['xp'][0]):
        raise ValueError('Problem in interpRBF, length of vector and rbf model do not match')
    ed = distLine(x, rbfModel['xp']) # euclidean distance of x to all xp, ed is vector of length nrow(xp)  
    
    ph = ed*ed*ed
    
    lhs = ph
    lhs = np.append(lhs,1)
    lhs = np.append(lhs, x)
    if squares:
        lhs = np.append(lhs, np.multiply(x,x))
    val = np.matmul(lhs,rbfModel['coef'])
    return val
   
def adjustMargins(Cfeas,Cinfeas,EPS,epsMax,dimension,feasible):
    
    if feasible:
        Cfeas += 1
        Cinfeas = 0
    else:
        Cinfeas += 1
        Cfeas = 0

    Tfeas = np.floor(2*np.sqrt(dimension)) # The threshhold parameter for the number of consecutive iterations that yield feasible solution before the margin is reduced
    Tinfeas = Tfeas # The threshold parameter for the number of consecutive iterations that yield infeasible solutions before the margin is increased
    
    if Cfeas>=Tfeas:
        EPS = EPS/2
        print('reducing epsilon to: '+str(EPS[0]))
        Cfeas = 0
    
    if Cinfeas >= Tinfeas:
        EPS = np.minimum(2*EPS,epsMax)
        print('increasing epsilon to: '+str(EPS[0]))
        Cinfeas = 0
        
    return(Cfeas, Cinfeas, EPS)
    
def predictConstraints(x, rbfmodel, EPS=None, rescale=True, drFactor=1):
#    print(x, 'constraints1')
    lb = rbfmodel['lb']
    ub = rbfmodel['ub']
    XI = rbfmodel['XI']
    xp = rbfmodel['xp']
    
    if rescale: #rescale, from lb an ub, to -1 and 1
        x = (2)*((x-lb)/(ub-lb))-1
    gamma = XI[ (len(xp) % len(XI) )]
    ro = gamma * 2
    
    x = np.matrix(x)
    
    distance = distLine(x, xp)
    subC = np.maximum(ro-distance,np.zeros(len(distance)))
    h = np.sum(subC)*drFactor
    constraintPrediction = interpolateRBF(x, rbfmodel)+EPS**2
    
    if np.any(np.isnan(constraintPrediction)):
#        raise Exception('predictConstraints: x value is NaN, returning Inf')
#        print(h, "constraints2")
        return(np.full(rbfmodel['coef'].shape[1]+1, -np.inf))
        
    h = np.append(np.array([-1*h]), -constraintPrediction)  
#    print(h, 'constraints3')
    return(h) 


#ax = plt.subplot(111)
#x = np.array([[-1],[0],[0.25],[0.5],[1]])
#y = []
#for xi in x:
#    y.append(xi**3+0.25)
#y = -1*np.array(y).reshape(len(x),1)
#ax.plot(x,y, 'ro',label='training points')
#
#rbfmodel = trainCubicRBF(x,y,-1,1,y)
#
#
#x = np.array(list(range(101))).reshape(101,1)
#x = x - 50
#x = x/50
#y = x**3+0.25
#y = -1*y
#ax.plot(x,y,'--',label='actual values')
#
#predicted = []
#for xi in x:
#    predicted.append(predictConstraints(xi,rbfmodel,0.01))
#predicted = np.array(predicted)
#predicted = -1*predicted[:,1]
#ax.plot(x,predicted,label='predicted values')
#
#ax.set_xlabel('x')
#ax.set_ylabel('g')
#ax.set_title('CRBF approximation')
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
#ax.axvspan(-1,-0.4,alpha=0.1,color='red',label='Infeasible area')
#plt.legend(bbox_to_anchor=(0.65, 1), loc=2, borderaxespad=0.,framealpha=0.)
#plt.savefig("RBFapproximation3.pdf",dpi=600,bbox_inches='tight') 




#ax = plt.subplot(111)
#
#x = np.array(list(range(101))).reshape(101,1)
#x = x - 50
#x = x/50
#y = x**2-0.5
#ax.plot(x,y,'--',label='Constraint function')
#
#ax.set_xlabel('x')
#ax.set_ylabel('g')
#ax.set_title('Constraint Example')
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
#ax.axvspan(-1,-0.707,alpha=0.1,color='red',label='Infeasible area')
#ax.axvspan(0.707,1,alpha=0.1,color='red')
#plt.legend(bbox_to_anchor=(0.3, 0.8), loc=2, borderaxespad=0.,framealpha=0.)
#plt.savefig("constraintExample.PNG",dpi=600,bbox_inches='tight') 