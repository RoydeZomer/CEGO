# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 10:14:58 2017

@author: r.dewinter
"""
import math
import numpy as np

def transformLHS(Design=None, lower=None, upper=None, stepsize=None):
    if lower is None:
        raise ValueError('Design, lower and upper are necessary input arguments')
    if Design is None:
        raise ValueError('Design, lower and upper are necessary input arguments')
    if upper is None:
        raise ValueError('Design, lower and upper are necessary input arguments')
    sizeLHS, no_param = Design.shape
    rangee = upper - lower
    if stepsize is None:
        stepsize = rangee/(2*sizeLHS)
    else:
        if len(stepsize) < no_param:
            stepsize = [stepsize[0]]*no_param
        for i in range(no_param):
            if stepsize[i] == 0:
                stepsize[i] = rangee[i]/(2*sizeLHS)
            else:
                lower[i] = math.ceil(lower[i]/stepsize[i])*stepsize[i]
                upper[i] = math.floor(upper[i]/stepsize[i])*stepsize[i]
                rangee[i] = upper[i]-lower[i]
    steps = rangee/stepsize+1
    transformedDesign = np.ceil( np.multiply( Design,([steps]*sizeLHS)) )-1
    transformedDesign = [lower]*sizeLHS + np.multiply(transformedDesign, [stepsize]*sizeLHS)
    
    return transformedDesign