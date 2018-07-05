# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:18:54 2018

@author: r.dewinter
"""

import numpy as np

def CTP1(x):
    x1 = x[0]
    x2 = x[1]
    
    f1 = x1
    f2 = (1+x2)*np.exp((-1*x1)/(1+x2))
    
    g1 = f2 / (0.858 * np.exp( -0.541 * f1 )) - 1
    g2 = f2 / (0.728 * np.exp( -0.285 * f1 )) - 1
    
    objectives = np.array([f1, f2])
    constraints = np.array([g1,g2])
    constraints = -1*constraints #transform for sacobra
    return np.array([objectives, constraints])
    
#iteration time 80.75800013542175
#14392.611999988556
#14419.184000015259
    
#iteration time 26.826000213623047
#2550.3870000839233
#2563.927999973297
    
#iteration time 20.795000076293945
#2715.882000207901
#2727.6260001659393