# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 09:19:42 2018

@author: r.dewinter
"""
import numpy as np

def OSY(x):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    x4 = x[3]
    x5 = x[4]
    x6 = x[5]
    
    f1 = -25*(x1-2)**2 - (x2-2)**2 - (x3-1)**2 - (x4-4)**2 - (x5-1)**2
    f2 = x1**2 + x2**2 + x3**2 + x4**2 + x5**2 + x6**2
    
    g1 = x1 + x2 - 2
    g2 = 6 - x1 - x2
    g3 = 2 - x2 + x1
    g4 = 2 - x1 + 3*x2
    g5 = 4 - (x3-3)**2 - x4
    g6 = (x5-3)**2 + x6 -4
    
    objectives = np.array([f1, f2])
    constraints = np.array([g1,g2,g3,g4,g5,g6])
    constraints = -1*constraints #transform for sacobra
    return np.array([objectives, constraints])