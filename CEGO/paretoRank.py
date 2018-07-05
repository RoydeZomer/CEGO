# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 13:52:20 2017

@author: r.dewinter
"""
from paretofrontFeasible import paretofrontFeasible 
import numpy as np

def paretoRank(objectives, constraints):
    nPop = len(objectives)
    ranks = np.zeros(nPop)
    popInd = np.array([True]*nPop)
    nPv = 1
    while sum(popInd)!=0:
        frontInd = popInd
        frontInd[popInd] = paretofrontFeasible(objectives[popInd,:], constraints[popInd,:])
        ranks[frontInd]  = nPv
        popInd = np.logical_xor(popInd, frontInd)
        nPv = nPv+1
    return ranks