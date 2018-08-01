# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 11:05:18 2017

@author: r.dewinter
"""

#conda install -c conda-forge pygmo

from CONSTRAINED_SMSEGO import CONSTRAINED_SMSEGO

from paretofrontFeasible import paretofrontFeasible

import numpy as np
import time
import pandas as pd
from functools import partial
import os

from hypervolume import hypervolume
from visualiseParetoFront import visualiseParetoFront

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import mpld3



def parallel_coordinates(parameters, constraints, objectives, outdir=None, parNames=None):
    feasible = np.sum(constraints<=0, axis=1)==constraints.shape[1]
    dominant = paretofrontFeasible(objectives, constraints)
    rank = feasible+0+dominant+0 #+0 to convert to int
    
    objectives = objectives *-1
    for i in range(objectives.shape[1]):
        objectives[:,i] = (objectives[:,i] - min(objectives[:,i])) / (max(objectives[:,i]) - min(objectives[:,i]))
        
    alpha = np.sum(objectives, axis=1)
    alpha = alpha/max(alpha)
    
    colors = np.empty((len(alpha),3))
    
    brightness2 = np.empty(len(alpha))
    brightness2[:] = 0
    idx = rank==2
    brightness2[idx] = np.array(range(sum(idx)))
    brightness2 = brightness2/2
    brightness2 = brightness2/sum(idx)
    colors[idx] = np.array([brightness2, [1]*len(brightness2), brightness2]).T[idx]
    
    brightness1 = np.empty(len(alpha))
    brightness1[:] = 0
    idx = rank==1
    brightness1[idx] = np.array(range(sum(idx)))
    brightness1 = brightness1/2
    brightness1 = brightness1/sum(idx)
    colors[idx] = np.array([brightness1, brightness1, [1]*len(brightness1)]).T[idx]
    
    brightness0 = np.empty(len(alpha))
    brightness0[:] = 0
    idx = rank==0
    brightness0[idx] = np.array(range(sum(idx)))
    brightness0 = brightness0/2
    brightness0 = brightness0/sum(idx)
    colors[idx] = np.array([[1]*len(brightness0), brightness0, brightness0]).T[idx]
    
    data = np.column_stack((rank, parameters))
    order = data[:,0].argsort()
    colors = colors[order]
    data = data[order] #sort on rank
    rank = data[:,0]
    data_sets = data[:,1:] #remove rank
    alpha = alpha[order]
    
    if parNames is None or len(parNames)!=parameters.shape[1]:
        columNames=['parameter'+str(i) for i in range(parameters.shape[1])]
    else:
        columNames = parNames
    
    dims = len(data_sets[0])
    x    = range(dims)
    fig, axes = plt.subplots(1, dims-1, sharey=False)

    if colors is None:
        colors = ['r-']*len(data_sets)
    
    # Calculate the limits on the data
    min_max_range = list()
    for m in zip(*data_sets):
        mn = min(m)
        mx = max(m)
        if mn == mx:
            mn -= 0.5
            mx = mn + 1.
        r  = float(mx - mn)
        min_max_range.append((mn, mx, r))

    # Normalize the data sets
    norm_data_sets = list()
    for ds in data_sets:
        nds = []
        for dimension, value in enumerate(ds):
            v = (value - min_max_range[dimension][0]) / min_max_range[dimension][2]
            nds.append(v)
        norm_data_sets.append(nds)
        
    data_sets = norm_data_sets

    # Plot the datasets on all the subplots
    for i, ax in enumerate(axes):
        for dsi, d in enumerate(data_sets):
            ax.plot(x, d, c=colors[dsi], alpha=alpha[dsi])
        ax.set_xlim([x[i], x[i+1]])
        
    # Set the x axis ticks 
    for dimension, (axx,xx) in enumerate(zip(axes, x[:-1])):
        axx.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ticks = len(axx.get_yticklabels())
        labels = list()
        step = min_max_range[dimension][2] / (ticks - 3)
        mn   = min_max_range[dimension][0]
        for i in range(-1,ticks):
            v = mn + i*step
            labels.append('%6.2f' % v) 
        axx.set_yticklabels(labels)


    # Move the final axis' ticks to the right-hand side
    axx = plt.twinx(axes[-1])
    dimension += 1
    axx.xaxis.set_major_locator(ticker.FixedLocator([x[-2], x[-1]]))
    ticks = len(axx.get_yticklabels())
    step = min_max_range[dimension][2] / (ticks - 1)
    mn   = min_max_range[dimension][0]
    labels = ['%6.2f' % (mn + i*step) for i in range(ticks)]
    axx.set_yticklabels(labels)      
    
    i=0
    for col in columNames[:-2]:
        plt.sca(axes[i])
        plt.xticks([i], (col,), rotation = 'vertical')
        i+=1
    plt.sca(axes[i])
    plt.xticks([i,i+1], columNames[i:],  rotation = 'vertical')
    
    #color labels
    plt.plot([],[],color='r',label='Infeasible')
    plt.plot([],[],color='b',label='Feasible')
    plt.plot([],[],color='g',label='Non-dominated')
    
    #delete whitespace
    plt.subplots_adjust(wspace=0)
    
    #title
    plt.suptitle('Parallel Coordinate Plot')
    
    plt.legend(bbox_to_anchor=(1.6, 1), loc=2, borderaxespad=0.)
    plt.show()
    if outdir is not None: 
        fig.savefig(str(outdir)+"paralelcoordinate1.pdf",dpi=600,bbox_inches='tight')
    else:
        fig.savefig("paralelcoordinate1.pdf",dpi=600,bbox_inches='tight')

def convergence_plot(objectives, constraints, ref, outdir=None):
    paretoOptimal = np.empty(len(objectives), dtype=bool)
    paretoOptimal[:] = False
    progress_hypervolume = np.empty(len(objectives))
    progress_hypervolume[:] = 0
    
    for i in range(len(progress_hypervolume)):
        paretoOptimal[:i] = paretofrontFeasible(objectives[:i,:],constraints[:i,:])
        paretoFront = objectives[paretoOptimal]
        currentHV = hypervolume(paretoFront, ref)
        progress_hypervolume[i] = currentHV
        
    plt.title('Convergence Plot')
    plt.xlabel('Iteration')
    plt.ylabel('(Hyper) Volume')
    plt.plot(range(len(progress_hypervolume)),progress_hypervolume)
    if outdir is not None:
        plt.savefig(str(outdir)+'ConvergencePlot.pdf',dpi=600)
    else:
        plt.savefig('ConvergencePlot.pdf',dpi=600)


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

    CONSTRAINED_SMSEGO_ORDER = np.append(objectiveValues, constraintValues)
    CONSTRAINED_SMSEGO_ORDER = CONSTRAINED_SMSEGO_ORDER.astype(float)
    
    print(CONSTRAINED_SMSEGO_ORDER)
    
    return(objectiveValues, CONSTRAINED_SMSEGO_ORDER[len(objectiveValues):])

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
tabReset = int(df_settings['VALUE']['TABRESET'])
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
functionName = str(problemCall).split(' ')[1]
outdir = 'results/'+str(functionName)+'/'

par_file_path = str(outdir)+'/par_run'+str(runNo)+'.csv'
con_file_path = str(outdir)+'/con_run'+str(runNo)+'.csv'
obj_file_path = str(outdir)+'/obj_run'+str(runNo)+'.csv'

if os.path.exists(par_file_path) and os.path.exists(con_file_path) and os.path.exists(obj_file_path) and (tabReset==0 or tabReset==2):
    parameters = np.genfromtxt(par_file_path, delimiter=',')
    constraints = np.genfromtxt(con_file_path, delimiter=',')
    objectives = np.genfromtxt(obj_file_path, delimiter=',')
    
    if tabReset==2:
        pf = paretofrontFeasible(objectives, constraints)
        parameters = parameters[pf]
        constraints = constraints[pf]
        objectives = objectives[pf]
    
    maxEval = max(initEval+len(parameters), maxEval)
    initEval = max(initEval, len(parameters)+6)
    
else:
    previousResults = np.genfromtxt('V:/temp/INITIAL_NAPA_RESULTS.csv', delimiter=',', skip_header=True)
    previousResults = np.asmatrix(previousResults)
    
    parameters = previousResults[:,3:3+nParameters]
    
    constraints = conMultiplier*np.array(previousResults[:,3+nParameters:3+nParameters+nConstraints] - conValues)
    objectives = objMultiplier*np.array(previousResults[:,3+nParameters+nConstraints:previousResults.shape[1]-1])

if not os.path.isdir(outdir):
    os.makedirs(outdir)

fileParameters = str(outdir)+'par_run'+str(runNo)+'_finalPF.csv'
fileObjectives = str(outdir)+'obj_run'+str(runNo)+'_finalPF.csv'
fileConstraints = str(outdir)+'con_run'+str(runNo)+'_finalPF.csv'

np.savetxt(fileParameters, parameters, delimiter=',')
np.savetxt(fileObjectives, objectives, delimiter=',')
np.savetxt(fileConstraints, constraints, delimiter=',')

s = time.time()
objectives, constraints, parameters = CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nConstraints, initEval, maxEval, smooth, runNo)
print(time.time()-s)



paretoOptimal = paretofrontFeasible(objectives,constraints)
paretoFront = objectives[paretoOptimal]

## save pareto frontier
objNames = list(df_objectives.index.values)
visualiseParetoFront(paretoFront, save=True, outdir=str(outdir), objNames=objNames)

## create convergence plot
convergence_plot(objectives, constraints, ref, outdir=str(outdir))

## create parallel coordinate plot
parNames = list(df_parameters.index.values)
parallel_coordinates(parameters, constraints, objectives, outdir=str(outdir), parNames=parNames)

