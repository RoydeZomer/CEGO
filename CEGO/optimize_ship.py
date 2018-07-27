# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:05:18 2017

@author: r.dewinter
"""
from CONSTRAINED_SMSEGO import CONSTRAINED_SMSEGO
from paretofrontFeasible import paretofrontFeasible

import numpy as np
import time
import pandas as pd
from functools import partial
import os


def compute_ship(x, nparameters, nconstraints, nobjectives, conValues, conMultiplier, objMultiplier):
    file_old = 'V:/temp/NAPA_RESULTS.csv'
    df_old_RESULTS = pd.read_csv(file_old)
    origheader = df_old_RESULTS.columns
    
    lenComputed = len(df_old_RESULTS)

    df_to_be_added = df_old_RESULTS.loc[lenComputed-1].values
    df_to_be_added[0] = df_to_be_added[0]+1
    df_to_be_added[1] = ''

    df_to_be_added[3:3+len(x)] = x
    df_to_be_added[3+len(x):len(df_to_be_added)] = np.zeros(len(df_to_be_added)-(3+len(x)))        
    
    df_old_RESULTS.loc[lenComputed] = df_to_be_added
    df_old_RESULTS = df_old_RESULTS[origheader]
    df_old_RESULTS = df_old_RESULTS.apply(pd.to_numeric, errors='ignore')

    file_to_compute = 'V:/temp/CEGO_PROPOSE.csv'
    df_old_RESULTS.to_csv(file_to_compute, sep=',', index=False)
    
    file_computed = 'V:/temp/NAPA_RESULTS.csv'
    df_new_RESULTS = pd.read_csv(file_computed)
    while lenComputed not in df_new_RESULTS.index or df_new_RESULTS['TARGET'][lenComputed]==0:
        try:
            df_new_RESULTS = pd.read_csv(file_computed)
            print('Read file')
            time.sleep(2)
        except OSError:
            print(OSError)
            time.sleep(2)
        
    result = df_new_RESULTS.loc[lenComputed].values
    objectiveValues = objMultiplier*result[3+len(x)+nconstraints:len(result)-1]
    
    constraintValues = conMultiplier*(result[3+len(x):3+len(x)+nconstraints] - conValues)
    
    print(objectiveValues)
    print(constraintValues)
    
#    CONSTRAINED_SMSEGO_ORDER = np.append( objectiveValues, constraintValues[:4]*-1+1)
#    CONSTRAINED_SMSEGO_ORDER = np.append(CONSTRAINED_SMSEGO_ORDER, -1*constraintValues[4:])
    CONSTRAINED_SMSEGO_ORDER = np.append(objectiveValues, constraintValues)
    CONSTRAINED_SMSEGO_ORDER = CONSTRAINED_SMSEGO_ORDER.astype(float)
    
    print(CONSTRAINED_SMSEGO_ORDER)
    
    return(objectiveValues, CONSTRAINED_SMSEGO_ORDER[2:])

#set cego parameters lowerlimit, upperlimit, number of constraints and reference point
file_old = 'V:/temp/INITIAL_NAPA_RESULTS.csv'
df_old_RESULTS = pd.read_csv(file_old)
file_to_compute = 'V:/temp/CEGO_PROPOSE.csv'
df_old_RESULTS.to_csv(file_to_compute, sep=',', index=False)
    
file = 'V:/temp/OBJECTIVES.csv'
df_objectives = pd.read_csv(file, index_col=0)
nObjectives = len(df_objectives)

file = 'V:/temp/CONSTRAINTS.csv'
df_constraints = pd.read_csv(file, index_col=0)
nConstraints = len(df_constraints)

file = 'V:/temp/PARAMETERS.csv'
df_parameters = pd.read_csv(file, index_col=0)
nParameters = len(df_parameters)

file = 'V:/temp/CEGO_SETTINGS.csv'
df_settings = pd.read_csv(file, index_col=0)

ranges = []
for var in df_parameters.iterrows():
    lolim = var[1]['LLIM']
    uplim = var[1]['ULIM']
    ranges.append([lolim, uplim])
ranges = np.array(ranges)

objRanges = []
objMultiplier = []
for var in df_objectives.iterrows():
    refPoint = var[1]['DES']
    if var[1]['GOAL']=='MAX':
        objMultiplier.append(-1)
        objRanges.append(-1*refPoint)
    else:
        objMultiplier.append(1)
        objRanges.append(refPoint)
ref = np.array(objRanges)
objMultiplier = np.array(objMultiplier)

conMultiplier = []
conValues = []
for var in df_constraints.iterrows():
    conValues.append(var[1]['VALUE'])
    if  var[1]['TYPE']=='=':
        conMultiplier.append(1)
    elif var[1]['TYPE']=='>':
        conMultiplier.append(-1)
    else:
        conMultiplier.append(1)
conMultiplier = np.array(conMultiplier)
conValues = np.array(conValues)

problemCall = partial(compute_ship, nparameters=nParameters, nconstraints=nConstraints, nobjectives=nObjectives, 
                      conValues=conValues, conMultiplier=conMultiplier, objMultiplier=objMultiplier)
rngMin = ranges[:,0]
rngMax = ranges[:,1]
initEval = int(df_settings['VALUE']['INITEVAL'])
maxEval = int(df_settings['VALUE']['LOOPS'])
smooth = int(df_settings['VALUE']['SMOOTH'])
runNo = int(df_settings['VALUE']['SEED'])
#ref = np.array([301,72])

if initEval == 0:
    initEval = 11*nParameters-1 

if maxEval < initEval:
    raise ValueError('Maximum number of iterations is smaller then initial number of Evaluations, CEGO terminates')

if initEval < nParameters+1:
    raise ValueError('Initial number of Evaluations to small, must at least be #parameters+1. Initial evaluations 11*#parameters-1 recommended. CEGO terminates')

if runNo == 0 :
    runNo = int(time.time())


###### read first or more previous results
previousResults = np.genfromtxt('V:/temp/INITIAL_NAPA_RESULTS.csv', delimiter=',', skip_header=True)
previousResults = np.asmatrix(previousResults)

parameters = previousResults[:,3:3+nParameters]
#constraints = previousResults[:,3+nParameters:3+nParameters+nConstraints]
#constraints[:,:4] = constraints[:,:4]*-1+1
#constraints[:,4:] = -1*constraints[:,4:]
#objectives = previousResults[:,3+nParameters+nConstraints:previousResults.shape[1]-1]

constraints = conMultiplier*np.array(previousResults[:,3+nParameters:3+nParameters+nConstraints] - conValues)
objectives = objMultiplier*np.array(previousResults[:,3+nParameters+nConstraints:previousResults.shape[1]-1])

functionName = str(problemCall).split(' ')[1]
outdir = 'results/'+str(functionName)+'/'

if not os.path.isdir(outdir):
    os.makedirs(outdir)

fileParameters = str(outdir)+'par_run'+str(runNo)+'_finalPF.csv'
fileObjectives = str(outdir)+'obj_run'+str(runNo)+'_finalPF.csv'
fileConstraints = str(outdir)+'con_run'+str(runNo)+'_finalPF.csv'

#paretoOptimal = paretofrontFeasible(objectives,constraints)
paretoOptimal = [True]*len(objectives)
paretoFront = objectives[paretoOptimal]
paretoSet = parameters[paretoOptimal]
paretoConstraints = constraints[paretoOptimal]

np.savetxt(fileParameters, paretoSet, delimiter=',')
np.savetxt(fileObjectives, paretoFront, delimiter=',')
np.savetxt(fileConstraints, paretoConstraints, delimiter=',')


s = time.time()
CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nConstraints, initEval, maxEval, smooth, runNo)
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