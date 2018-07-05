# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 17:13:58 2017

@author: r.dewinter
"""
import numpy as np

def corrnoisygauss(theta,d):
    m,n = d.shape
    lt = len(theta)
    if lt != n+1:
        raise ValueError('Length of theta must be 2*n+1') 
    
    poww = np.ones((m,n))+1
    
    tt = np.tile(-theta[:n],(m,1))
        
    td = np.multiply(np.power(np.abs(d),poww),tt)
    
    r = np.prod(np.exp(td),1)

    dr = poww * tt * np.sign(d) * np.power(np.abs(d), np.ones((m,n))) * np.tile(r,(n,1)).T
    
    return [r,dr]
    