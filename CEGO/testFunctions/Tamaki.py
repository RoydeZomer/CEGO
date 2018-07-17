# -*- coding: utf-8 -*-
"""
Created on Sun Mar  4 15:53:28 2018

@author: r.dewinter
"""
import numpy as np


def Tamaki(x):
    f1 = -x[0]
    f2 = -x[1]
    f3 = -x[2]
    
    g1 = np.sum(x**2)-3
    
    return np.array([f1,f2,f3]), -1*np.array([g1])

