# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:01:43 2018

@author: r.dewinter
"""

import numpy as np
import numpy.ctypeslib as npct
from ctypes import c_int
from ctypes import c_double

##########################################################################
# define input/output type
array_1d_double = npct.ndpointer(dtype=np.double, flags='C_CONTIGUOUS')
array_1d_bool = npct.ndpointer(dtype=np.bool,  flags='C_CONTIGUOUS')
##########################################################################
# load the libraries, using numpy mechanisms : use the name *.so

libgd  = npct.load_library("libgd.so",".")

# 4. GD library (GD,IGD)
libgd.gd.restype = c_double
libgd.gd.argtypes = [array_1d_double, array_1d_double, c_int, c_int, c_int]
libgd.igd.restype = c_double
libgd.igd.argtypes = [array_1d_double, array_1d_double, c_int, c_int, c_int]
libgd.incr_gd.restype = None
libgd.incr_gd.argtypes = [array_1d_double,array_1d_double, array_1d_double, c_int, c_int, c_int]
libgd.incr_igd.restype = None
libgd.incr_igd.argtypes = [array_1d_double,array_1d_double, array_1d_double, c_int, c_int, c_int]



def compute_gd(approximation_set, reference_set):
	"""
		returns the generational distance indicator value of the approximation set with respect to the reference set
	"""
	#assert approximation_set.shape[1] == reference_set.shape[1]
	return libgd.gd(approximation_set, reference_set, approximation_set.shape[0], reference_set.shape[0], approximation_set.shape[1])
