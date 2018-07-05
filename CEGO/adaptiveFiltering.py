# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 17:46:51 2017

@author: r.dewinter
"""
import numpy as np

def adaptiveFiltering(fullSet=None, gridSet=None, score=None):
    if fullSet is None or gridSet is None:
        raise ValueError('fullSet and gridSet are necessary input arguments')
    Nf, n = fullSet.shape
    Ng = len(gridSet)
    if score is None:
        score = list(range(Nf))
    
    ix = np.zeros((Nf,n))
    accept = np.full((Nf), True, dtype=bool)
    cells = np.sort(gridSet, axis=None).reshape(gridSet.shape)
    
    for i in range(Nf):
        for k in range(n):
            pos = np.where(cells[:,k]>fullSet[i,k])[0]
            if len(pos)==0:
                ix[i,k] = Ng
            else:
                ix[i,k] = pos[0]
    
    
    for i in range(Nf-1):
        diffPar = ~np.any(np.abs(ix[i,:]-ix), axis=1)
        
        scoreBool = score[i] > score[i+1:Nf]
        if np.any(np.logical_and(diffPar[i+1:], scoreBool)):
            accept[i] = False
            
        logicBool = np.logical_and(diffPar[i+1:], ~scoreBool)
        accept[i+1:Nf][logicBool] = False
        if np.all(~accept[i:]):
            break
            
#    s = time.time()
#    for i in range(Nf-1):
#            for j in range(i+1,Nf):            
#                diffPar = abs(ix[i,:]-ix[j,:])
#                if not any(diffPar):
#                    if score[i] < score[j]:
#                        accept[j] = False
#                    else:
#                        accept[i] = False
#    e = time.time()
                    
    if accept.size==0:
        return([np.array([]),np.array([])])
    filteredSet = fullSet[accept]
    return [filteredSet, accept]