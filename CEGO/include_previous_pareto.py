# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 11:26:32 2017

@author: r.dewinter
"""
import numpy as np


def include_previous_pareto(initEval=None, outdir=None, runNo=0):
    if initEval is None:
        raise ValueError('InitEvalLHS must be set')
    if outdir is None:
        raise ValueError('outdir must be set')
    
    fileParameters = str(outdir)+'par_run'+str(runNo)+'_finalPF.csv'
    fileObjectives = str(outdir)+'obj_run'+str(runNo)+'_finalPF.csv'
    fileConstraints = str(outdir)+'con_run'+str(runNo)+'_finalPF.csv'
    
    par_old = np.genfromtxt(fileParameters, delimiter=',')
    obj_old = np.genfromtxt(fileObjectives, delimiter=',')
    con_old = np.genfromtxt(fileConstraints, delimiter=',') 
    
    if par_old.ndim == 1:
        par_old = np.array([par_old])
        obj_old = np.array([obj_old])
        con_old = np.array([con_old])
    
    if len(par_old)>initEval: 
        return par_old[:initEval, :], con_old[:initEval, :], obj_old[:initEval, :]
    else:
        return par_old, con_old, obj_old