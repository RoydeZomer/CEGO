# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 14:40:04 2017

@author: r.dewinter
"""
from testFunctions.TBTD import TBTD
from testFunctions.SRD import SRD
from testFunctions.WB import WB
from testFunctions.DBD import DBD
from testFunctions.SPD import SPD
from testFunctions.CSI import CSI
from testFunctions.WP import WP

from testFunctions.OSY import OSY
from testFunctions.CTP1 import CTP1
from testFunctions.CEXP import CEXP
from testFunctions.C3DTLZ4 import C3DTLZ4
from testFunctions.TNK import TNK
from testFunctions.SRN import SRN
from testFunctions.BNH import BNH
from CONSTRAINED_SMSEGO import CONSTRAINED_SMSEGO
import time

import numpy as np

## Real world like problems 

problemCall = CSI
rngMin = np.array([0.5,    0.45,  0.5,  0.5,   0.875,     0.4,    0.4])
rngMax = np.array([1.5,    1.35,  1.5,  1.5,   2.625,     1.2,    1.2])
initEval = 30
maxEval = 200
smooth = 2
nVar = 7
runNo = 11
ref = np.array([42,4.5,13])
nconstraints = 10

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)

problemCall = WB
rngMin = np.array([0.125, 0.1, 0.1, 0.125])
rngMax = np.array([5, 10, 10, 5])
initEval = 30
maxEval = 200
smooth = 2
nVar = 4
runNo = 3
ref = np.array([350,0.1])
nconstraints = 5

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)

problemCall = TBTD
rngMin = np.array([1,0.0005,0.0005])
rngMax = np.array([3,0.05,0.05])
initEval = 30
maxEval = 200
smooth = 2
runNo = 5
ref = np.array([0.1,100000])
nconstraints = 3

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)

problemCall = DBD
rngMin = np.array([55, 75, 1000, 2])
rngMax = np.array([80, 110, 3000, 20])
initEval = 30
maxEval = 200
smooth = 2
nVar = 4
runNo = 3
ref = np.array([5,50])
nconstraints = 5

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)

problemCall = SPD
rngMin = np.array([150,    25,    12,   8,     14, 0.63])
rngMax = np.array([274.32, 32.31, 22,   11.71, 18, 0.75])
initEval = 30
maxEval = 200
smooth = 2
nVar = 6
runNo = 5
ref = np.array([16,19000,-260000])
nconstraints=9

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)


problemCall = WP
rngMin = np.array([0.01,    0.01,  0.01])
rngMax = np.array([0.45,    0.1,  0.1])
initEval = 30
maxEval = 200
smooth = 2
nVar = 3
runNo = 3
ref = np.array([83000, 1350, 2.85, 15989825, 25000])
nconstraints = 7

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)

##########################theoreticala problems

problemCall = BNH
rngMin = np.array([0,0])
rngMax = np.array([5,3])
initEval = 30
maxEval = 200
smooth = 2
runNo = 2
ref = np.array([140,50])
nconstraints = 2

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)

problemCall = SRN
rngMin = np.array([-20,-20])
rngMax = np.array([20, 20])
initEval = 30
maxEval = 200
smooth = 2
runNo = 8
ref = np.array([301,72])
nconstraints = 2

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)

problemCall = TNK
rngMin = np.array([1e-5,1e-5])
rngMax = np.array([np.pi, np.pi])
initEval = 30
maxEval = 200
smooth = 2
runNo = 2
ref = np.array([3,3])
nconstraints = 2

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)


problemCall = C3DTLZ4
rngMin = np.array([0,0,0,0,0,0])
rngMax = np.array([1,1,1,1,1,1])
initEval = 30
maxEval = 200
smooth = 2
runNo = 8
ref = np.array([3,3])
nconstraints = 2

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)


problemCall = CTP1
rngMin = np.array([0,0])
rngMax = np.array([1,1])
initEval = 21
maxEval = 200
smooth = 2
runNo = 5
ref = np.array([1,2])
nconstraints = 2

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)


problemCall = CEXP
rngMin = np.array([0.1,0])
rngMax = np.array([1,5])
initEval = 21
maxEval = 200
smooth = 2
runNo = 6
ref = np.array([1,9])
nconstraints = 2

epsilonInit=0.01
epsilonMax=0.02
s = time.time()
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)


problemCall = OSY
rngMin = np.array([0,0,1,0,1,0])
rngMax = np.array([10,10,5,6,5,10])
initEval = 30
maxEval = 200
smooth = 2
runNo = 5
ref = np.array([0,386])
nconstraints = 6

epsilonInit=0.01
epsilonMax=0.02
s = time.time() 
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nconstraints)
print(time.time()-s)

#par = len(rngMin)
#obj = len(ref)
#runs = 1000000
#objectives = np.empty((runs,obj))
#constraints = np.empty((runs,nconstraints))
#for i in range(runs):
#    x = rngMax*np.random.rand(par)+rngMin
#    objectives[i],constraints[i] = problemCall(x)
#np.sum(np.sum(constraints<0,axis=1)==nconstraints)/runs