# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:05:18 2017

@author: r.dewinter
"""
from compute_ship import compute_ship
from CONSTRAINED_SMSEGO import CONSTRAINED_SMSEGO
from paretofrontFeasible import paretofrontFeasible


import numpy as np
import time
import pandas as pd
from functools import partial
import os

#set cego parameters lowerlimit, upperlimit, number of constraints and reference point
nConstraints = 16
file = 'V:/temp/FUNCT_SACOBRA_TUNING.csv'
df_parameters = pd.read_csv(file, index_col=0)

file = 'V:/temp/OPT_PARAMETER.csv'
df_limits = pd.read_csv(file, index_col=0)

ranges = []
decimalPoints = []
for var in df_limits.iterrows():
    lolim = var[1]['LLIM']
    uplim = var[1]['ULIM']
    ranges.append([lolim, uplim])

ranges = np.array(ranges)

problemCall = partial(compute_ship, nconstraints=nConstraints)
rngMin = ranges[:,0]
rngMax = ranges[:,1]
initEval = 30
maxEval = int(df_parameters['VALUE']['LOOPS'])
smooth = 2
runNo = 300
ref = np.array([5000,2])


###### read first or more previous results
previousResults = np.genfromtxt('V:/temp/INITIAL_OPT_SACOBRA_RESULTS.csv', delimiter=',', skip_header=True)
previousResults = np.asmatrix(previousResults)
decisionVariables = len(ranges)

parameters = previousResults[:,3:3+decisionVariables]
constraints = previousResults[:,3+decisionVariables:3+decisionVariables+nConstraints]
constraints[:,:4] = constraints[:,:4]*-1+1
constraints[:,4:] = -1*constraints[:,4:]
objectives = previousResults[:,3+decisionVariables+nConstraints:previousResults.shape[1]-1]

functionName = str(problemCall).split(' ')[1]
outdir = 'results/'+str(functionName)+'/'

if not os.path.isdir(outdir):
    os.makedirs(outdir)

fileParameters = str(outdir)+'par_run'+str(runNo)+'_finalPF.csv'
fileObjectives = str(outdir)+'obj_run'+str(runNo)+'_finalPF.csv'
fileConstraints = str(outdir)+'con_run'+str(runNo)+'_finalPF.csv'

#paretoOptimal = paretofrontFeasible(objectives,constraints)
paretoOptimal = [True]
paretoFront = objectives[paretoOptimal]
paretoSet = parameters[paretoOptimal]
paretoConstraints = constraints[paretoOptimal]

np.savetxt(fileParameters, paretoSet, delimiter=',')
np.savetxt(fileObjectives, paretoFront, delimiter=',')
np.savetxt(fileConstraints, paretoConstraints, delimiter=',')

#start CEGO!
s = time.time()
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, initEval, maxEval, smooth, runNo, ref, nConstraints)
print(time.time()-s)

#7uur 55 minuten 35 seconden

#pf = np.array([[  2.54336000e+03   ,8.47891000e-01],
# [  2.49148000e+03,   8.52148000e-01],
# [  2.76051000e+03 ,  8.44227000e-01],
# [  2.29118000e+03  , 8.53175000e-01],
# [  1.72588000e+03   ,1.20782000e+00],
# [  1.92200000e+03,   8.58907000e-01],
# [  2.85871000e+03 ,  8.05092000e-01],
# [  2.06695000e+03  , 8.53343000e-01],
# [  1.74834000e+03   ,8.76522000e-01],
# [  1.79203000e+03   ,8.70181000e-01]])