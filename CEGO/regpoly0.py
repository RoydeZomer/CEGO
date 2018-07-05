# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:51:00 2017

@author: r.dewinter
"""
import numpy as np

def regpoly0(S):
    m,n = S.shape
    f = np.ones((m,1))
    df = np.zeros(n)
    return [f, df]