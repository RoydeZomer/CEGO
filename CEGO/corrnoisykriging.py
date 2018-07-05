# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:55:02 2017

@author: r.dewinter
"""
import numpy as np

def corrnoisykriging(theta, d):
    m,n = d.shape
    lt = len(theta)
    if lt != 2*n+1:
        raise ValueError('Length of theta must be 2*n+1') 
        
    poww = theta[n:2*n]
    tt = -theta[:n]
    nugget = theta[2*n]
    td = np.multiply( np.power(np.abs(d), poww) ,tt)
    r = nugget * np.prod(np.exp(td),1)
    dr = nugget*poww*tt*np.sign(d)*np.power(np.abs(d), poww-np.ones((m,n))) * np.reshape(np.repeat(r,n), d.shape)
    return [r,dr]