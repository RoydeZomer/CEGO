# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 10:22:20 2018

@author: r.dewinter
"""
import pandas as pd
import numpy as np
import time
#import matplotlib.pyplot as plt 
#i+=1
#x = parameters[i,:]

def compute_ship(x, nconstraints):
    file_old = 'V:/temp/FUNCT_OPT_SACOBRA_RESULTS.csv'
    df_old_RESULTS = pd.read_csv(file_old)
    origheader = df_old_RESULTS.columns
    
    lenComputed = len(df_old_RESULTS)
    if lenComputed > 0:
        df_to_be_added = df_old_RESULTS.loc[lenComputed-1].values
        df_to_be_added[0] = df_to_be_added[0]+1
        df_to_be_added[1] = ''
    else:
        #hardcoded constraints 16x0 + objectives 2x0 + target 1x0
        df_to_be_added = np.array([1,'ORIG','N', 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 ])
    df_to_be_added[3:3+len(x)] = x
    df_to_be_added[3+len(x):len(df_to_be_added)] = np.zeros(len(df_to_be_added)-(3+len(x)))        
    
    df_old_RESULTS.loc[lenComputed] = df_to_be_added
    df_old_RESULTS = df_old_RESULTS[origheader]
    df_old_RESULTS = df_old_RESULTS.apply(pd.to_numeric, errors='ignore')

    file_to_compute = 'V:/temp/NEW_GEN_OPT_SACOBRA_RESULTS.csv'
    df_old_RESULTS.to_csv(file_to_compute, sep=',', index=False)
    
    file_computed = 'V:/temp/FUNCT_OPT_SACOBRA_RESULTS.csv'
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
    objectiveValues = result[3+len(x)+nconstraints:len(result)-1]
    constraintValues = result[3+len(x):3+len(x)+nconstraints]
    
    print(objectiveValues)
    print(constraintValues)
    
    CONSTRAINED_SMSEGO_ORDER = np.append( objectiveValues, constraintValues[:4]*-1+1)
    CONSTRAINED_SMSEGO_ORDER = np.append(CONSTRAINED_SMSEGO_ORDER, -1*constraintValues[4:])
    CONSTRAINED_SMSEGO_ORDER = CONSTRAINED_SMSEGO_ORDER.astype(float)
    
    print(CONSTRAINED_SMSEGO_ORDER)
    
    return(objectiveValues, CONSTRAINED_SMSEGO_ORDER[2:])
    
#rngMin = np.array([5,16,5,12,-2.8,-1.6]) 
#rngMax = np.array([9,22,9,16,9.8,3.4])
#par = len(rngMin)
#obj = 2
#runs = 200
#nconstraints=16
#objectives = np.empty((runs,obj))
#constraints = np.empty((runs,nconstraints))
#for i in range(runs):
#    x = rngMax*np.random.rand(par)+rngMin
#    objectives[i],constraints[i] = problemCall(x)
#np.sum(np.sum(constraints<0,axis=1)==nconstraints)/runs