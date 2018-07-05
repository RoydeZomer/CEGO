# -*- coding: utf-8 -*-
"""
Created on Fri Apr 20 13:18:14 2018

@author: r.dewinter
"""

import numpy as np
from hypervolume import hypervolume
from paretofrontFeasible import paretofrontFeasible

from scipy.stats import ttest_ind
import pandas as pd
import os

#ttest
CEGOFolder = 'CONSTRAINED_EGO/results/'
NSGAFolder = 'NSGAII/'
SPEAFolder = 'SPEA2/'
MOGAFolder = 'MOGA/'

functions = ['BNH','C3DTLZ4','CEXP','CSI','CTP1','DBD','OSY','SPD','SRD','SRN','TBTD','TNK','WB','WP']
refs = [[140,50],[3,3],[1,9],[42,4.5,13],[1,2],[5,50],[0,386],[16,19000,-260000],[7000,1700],[301,72],[0.1,100000],[3,3],[350,0.1],[83000, 1350, 2.85, 15989825, 25000]]
refPoints = []

i = 0
for funcName in functions:
    hypCEGO = []
    hypNSGA = []
    hypSPEA = []
    hypMOGA = []
    ref = np.array(refs[i])
    
    
    for run in range(100):
        file = str(NSGAFolder)+'/'+str(funcName)+'/'+str(funcName)+'_pf_run_'+str(run)+'.csv'
        objectivesNSGAII = np.genfromtxt(file, delimiter=',')
        try:
            hypNSGA.append(hypervolume(objectivesNSGAII, ref))
        except:
            hypNSGA.append(hypervolume(np.array([objectivesNSGAII,objectivesNSGAII]), ref))
    
    for run in range(100):
        file = str(SPEAFolder)+'/'+str(funcName)+'/'+str(funcName)+'_pf_run_'+str(run)+'.csv'
        objectivesSPEA2 = np.genfromtxt(file, delimiter=',')
        try:
            hypSPEA.append(hypervolume(objectivesSPEA2, ref))
        except:
            hypSPEA.append(hypervolume(np.array([objectivesSPEA2,objectivesSPEA2]), ref))
        
    for run in range(1,11):
        file = str(MOGAFolder)+'/'+str(funcName)+'/'+str(funcName)+str(run)+'.csv'
        mogadata = pd.read_csv(file)
        colnames = list(mogadata.columns)
        objCols = [name for name in colnames if 'O_' in name]
        objectivesMOGA = mogadata[objCols].values
        conCols = [name for name in colnames if 'C_' in name]
        constraintMOGA = mogadata[conCols].values
        pff = paretofrontFeasible(objectivesMOGA, constraintMOGA)
        hypMOGA.append(hypervolume(objectivesMOGA[pff],ref))
        
    cegoFolderFunction = str(CEGOFolder)+'/'+str(funcName)+'/'
    for run in os.listdir(cegoFolderFunction):
        if 'run' in run:
            file = cegoFolderFunction+run+'/obj_'+str(run)+'_finalPF.csv'
            objectivesCEGO = np.genfromtxt(file, delimiter=',')
            hypCEGO.append(hypervolume(objectivesCEGO, ref))
    
    threshold = 0.01

    #print(funcName)
    print(funcName,'&',np.mean(hypNSGA),'&',np.mean(hypSPEA),'&',np.mean(hypMOGA),'&',np.mean(hypCEGO),'\\')
    print(funcName,'&',np.std(hypNSGA),'&',np.std(hypSPEA),'&',np.std(hypMOGA),'&',np.std(hypCEGO),'\\')
    if(ttest_ind(hypCEGO,hypNSGA,equal_var=False).pvalue < threshold):
        print('CEGO and NSGAII do not have an equal avarages')
    else:
        print('CEGO is not better then NSGAII')
    if(ttest_ind(hypCEGO,hypSPEA,equal_var=False).pvalue < threshold):
        print('CEGO and SPEA22 do not have an equal avarages')
    else:
        print('CEGO is not better then SPEA22')
    if(ttest_ind(hypCEGO,hypMOGA,equal_var=False).pvalue < threshold):
        print('CEGO and MOGAAA do not have an equal avarages')
    else:
        print('CEGO is not better then MOGAAA')        
    i+=1

functions = ['compute_ship']
refs = [[5000,2]]
refPoints = []

i = 0
for funcName in functions:
    hypCEGO = []
    hypNSGA = []
    hypSPEA = []
    hypMOGA = []
    ref = np.array(refs[i])
    
    for run in range(1,6):
        file = str(NSGAFolder)+'/'+str(funcName)+'/run'+str(run)+'/FUNCT_OPT_SACOBRA_RESULTS.csv'
        dataNSGAII = np.genfromtxt(file, delimiter=',',skip_header=True)
        constraintsNSGAII = dataNSGAII[:,-19:-3]
        constraintsNSGAII[:,:4] = constraintsNSGAII[:,:4]-1
        constraintsNSGAII = -1*constraintsNSGAII
        objectivesNSGAII = dataNSGAII[:,-3:-1]
        pff = paretofrontFeasible(objectivesNSGAII, constraintsNSGAII)
        hypNSGA.append(hypervolume(objectivesNSGAII[pff], ref))
    
    for run in range(1,6):
        file = str(SPEAFolder)+'/'+str(funcName)+'/run'+str(run)+'/FUNCT_OPT_SACOBRA_RESULTS.csv'
        dataSPEA = np.genfromtxt(file, delimiter=',',skip_header=True)
        constraintsSPEA = dataSPEA[:,-19:-3]
        constraintsSPEA[:,:4] = constraintsSPEA[:,:4]-1
        constraintsSPEA = -1*constraintsSPEA
        objectivesSPEA = dataSPEA[:,-3:-1]
        pff = paretofrontFeasible(objectivesSPEA, constraintsSPEA)
        hypSPEA.append(hypervolume(objectivesSPEA[pff], ref))
        
    for run in range(1,6):
        file = str(MOGAFolder)+'/'+str(funcName)+'/moga'+str(run)+'.csv'
        dataMOGA = np.genfromtxt(file, delimiter=',',skip_header=True)
        constraintsMOGA = dataMOGA[:,-19:-3]
        constraintsMOGA[:,:4] = constraintsMOGA[:,:4]-1
        constraintsMOGA = -1*constraintsMOGA
        objectivesMOGA = dataMOGA[:,-3:-1]
        pff = paretofrontFeasible(objectivesMOGA, constraintsMOGA)
        hypMOGA.append(hypervolume(objectivesMOGA[pff], ref))
        
    cegoFolderFunction = str(CEGOFolder)+'/'+str(funcName)+'/'
    for run in os.listdir(cegoFolderFunction):
        if run in ['run1', 'run2', 'run3', 'run4']:
            file = cegoFolderFunction+run+'/obj_'+str(run)+'_finalPF.csv'
            objectivesCEGO = np.genfromtxt(file, delimiter=',')
            hypCEGO.append(hypervolume(objectivesCEGO, ref))
    
    threshold = 0.05

    #print(funcName)
    print(funcName,'&',np.mean(hypNSGA),'&',np.mean(hypSPEA),'&',np.mean(hypMOGA),'&',np.mean(hypCEGO),'\\')
    print(funcName,'&',np.std(hypNSGA),'&',np.std(hypSPEA),'&',np.std(hypMOGA),'&',np.std(hypCEGO),'\\')
    if(ttest_ind(hypCEGO,hypNSGA,equal_var=False).pvalue < threshold):
        print('CEGO and NSGAII do not have an equal avarages')
    else:
        print('CEGO is not better then NSGAII')
    if(ttest_ind(hypCEGO,hypSPEA,equal_var=False).pvalue < threshold):
        print('CEGO and SPEA22 do not have an equal avarages')
    else:
        print('CEGO is not better then SPEA22')
    if(ttest_ind(hypCEGO,hypMOGA,equal_var=False).pvalue < threshold):
        print('CEGO and MOGAAA do not have an equal avarages')
    else:
        print('CEGO is not better then MOGAAA')       
    i+=1
    