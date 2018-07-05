# -*- coding: utf-8 -*-
"""
Created on Fri Mar 16 12:17:20 2018

@author: r.dewinter
"""

import pygmo as pg
import numpy as np

def hypervolume(pointset, ref):
    """Compute the absolute hypervolume of a *pointset* according to the
    reference point *ref*.
    """
    
    #make sure all points are smaller then pointset
    pointset = pointset[np.all(pointset<=ref,axis=1)]
    
    if len(pointset)==0:
        return 0
    hv = pg.hypervolume(pointset)
    contribution = hv.compute(ref)
    return contribution