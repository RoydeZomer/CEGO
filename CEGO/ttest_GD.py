# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 17:21:03 2018

@author: r.dewinter
"""

from TBTD import TBTD
from SRD import SRD
from WB import WB
from DBD import DBD
from SPD import SPD
from CSI import CSI
from WP import WP

#from DTLZ8 import DTLZ8
from OSY import OSY
from CTP1 import CTP1
from CEXP import CEXP
from C3DTLZ4 import C3DTLZ4
from TNK import TNK
from SRN import SRN
from BNH import BNH

#from compute_ship import compute_ship

import numpy as np
from platypus import NSGAII
from platypus import Problem
from platypus import Real
from platypus import nondominated
from compute_gd import compute_gd
import ast
import os

import numpy as np
from paretofrontFeasible import paretofrontFeasible

from scipy.stats import ttest_ind
import pandas as pd
import os


CEGOFolder = 'CONSTRAINED_EGO/results/'
NSGAFolder = 'NSGAII/'
SPEAFolder = 'SPEA2/'
MOGAFolder = 'MOGA/'

functions = ['BNH','C3DTLZ4','CEXP','CSI','CTP1','DBD','OSY','SPD','SRD','SRN','TBTD','TNK','WB','WP']

def doMeasurement(refPF, funcName, ref):    
    gdCEGO = []
    gdNSGA = []
    gdSPEA = []
    gdMOGA = []    
    
    ref = np.array([ref])
    ref = ref/1.0 #make it a float
    refPF = refPF/ref #normalise
    for run in range(100):
        file = str(NSGAFolder)+'/'+str(funcName)+'/'+str(funcName)+'_pf_run_'+str(run)+'.csv'
        objectivesNSGAII = np.genfromtxt(file, delimiter=',')
        if len(objectivesNSGAII)==0:
            gdNSGA.append(compute_gd(ref/ref,refPF))
        else:
            objectivesNSGAII = objectivesNSGAII/ref
            try:
                gdNSGA.append(compute_gd(objectivesNSGAII,refPF))
            except:
                gdNSGA.append(compute_gd(np.array([objectivesNSGAII,objectivesNSGAII]),refPF))
    
    for run in range(100):
        file = str(SPEAFolder)+'/'+str(funcName)+'/'+str(funcName)+'_pf_run_'+str(run)+'.csv'
        objectivesSPEA2 = np.genfromtxt(file, delimiter=',')
        if len(objectivesSPEA2)==0:
            gdSPEA.append(compute_gd(ref/ref,refPF))
        else:
            objectivesSPEA2 = objectivesSPEA2/ref
            try:
                gdSPEA.append(compute_gd(objectivesSPEA2,refPF))
            except:
                gdSPEA.append(compute_gd(np.array([objectivesSPEA2,objectivesSPEA2]),refPF))
        
    for run in range(1,11):
        file = str(MOGAFolder)+'/'+str(funcName)+'/'+str(funcName)+str(run)+'.csv'
        mogadata = pd.read_csv(file)
        colnames = list(mogadata.columns)
        objCols = [name for name in colnames if 'O_' in name]
        objectivesMOGA = mogadata[objCols].values
        conCols = [name for name in colnames if 'C_' in name]
        constraintMOGA = mogadata[conCols].values
        pff = paretofrontFeasible(objectivesMOGA, constraintMOGA)
        print(len(objectivesMOGA[pff]))
        if len(objectivesMOGA[pff])==0:
            gdMOGA.append(compute_gd(ref/ref,refPF))
        else: 
            objectivesMOGA = objectivesMOGA/ref
            gdMOGA.append(compute_gd(objectivesMOGA[pff],refPF))
        
    cegoFolderFunction = str(CEGOFolder)+'/'+str(funcName)+'/'
    for run in os.listdir(cegoFolderFunction):
        if 'run' in run:
            file = cegoFolderFunction+run+'/obj_'+str(run)+'_finalPF.csv'
            objectivesCEGO = np.genfromtxt(file, delimiter=',')
            if len(objectivesCEGO)==0:
                gdCEGO.append(compute_gd(ref/ref, refPF))
            else:
                objectivesCEGO = objectivesCEGO/ref
                gdCEGO.append(compute_gd(objectivesCEGO, refPF))
    
    print('nsga',funcName,np.mean(gdNSGA))
    print('spea',funcName,np.mean(gdSPEA))
    print('moga',funcName,np.mean(gdMOGA))
    print('cego',funcName,np.mean(gdCEGO))
    
def doMeasurementsOSY(refPF, funcName, ref):
    ref[0] = 300
    
    gdCEGO = []
    gdNSGA = []
    gdSPEA = []
    gdMOGA = []    
    
    ref = np.array([ref])
    ref = ref/1.0 #make it a float
    refPF = refPF/ref #normalise
    for run in range(100):
        file = str(NSGAFolder)+'/'+str(funcName)+'/'+str(funcName)+'_pf_run_'+str(run)+'.csv'
        objectivesNSGAII = np.genfromtxt(file, delimiter=',')
        if len(objectivesNSGAII)==0:
            gdNSGA.append(compute_gd(ref/ref,refPF))
        else:
            objectivesNSGAII = objectivesNSGAII/ref
            try:
                gdNSGA.append(compute_gd(objectivesNSGAII,refPF))
            except:
                gdNSGA.append(compute_gd(np.array([objectivesNSGAII,objectivesNSGAII]),refPF))
    
    for run in range(100):
        file = str(SPEAFolder)+'/'+str(funcName)+'/'+str(funcName)+'_pf_run_'+str(run)+'.csv'
        objectivesSPEA2 = np.genfromtxt(file, delimiter=',')
        if len(objectivesSPEA2)==0:
            gdSPEA.append(compute_gd(ref/ref,refPF))
        else:
            objectivesSPEA2 = objectivesSPEA2/ref
            try:
                gdSPEA.append(compute_gd(objectivesSPEA2,refPF))
            except:
                gdSPEA.append(compute_gd(np.array([objectivesSPEA2,objectivesSPEA2]),refPF))
        
    for run in range(1,11):
        file = str(MOGAFolder)+'/'+str(funcName)+'/'+str(funcName)+str(run)+'.csv'
        mogadata = pd.read_csv(file)
        colnames = list(mogadata.columns)
        objCols = [name for name in colnames if 'O_' in name]
        objectivesMOGA = mogadata[objCols].values
        conCols = [name for name in colnames if 'C_' in name]
        constraintMOGA = mogadata[conCols].values
        pff = paretofrontFeasible(objectivesMOGA, -1*constraintMOGA)
        print(len(objectivesMOGA[pff]))
        if len(objectivesMOGA[pff])==0:
            gdMOGA.append(compute_gd(ref/ref,refPF))
        else: 
            objectivesMOGA = objectivesMOGA/ref
            gdMOGA.append(compute_gd(objectivesMOGA[pff],refPF))
        
    cegoFolderFunction = str(CEGOFolder)+'/'+str(funcName)+'/'
    for run in os.listdir(cegoFolderFunction):
        if 'run' in run:
            file = cegoFolderFunction+run+'/obj_'+str(run)+'_finalPF.csv'
            objectivesCEGO = np.genfromtxt(file, delimiter=',')
            if len(objectivesCEGO)==0:
                gdCEGO.append(compute_gd(ref/ref, refPF))
            else:
                objectivesCEGO = objectivesCEGO/ref
                gdCEGO.append(compute_gd(objectivesCEGO, refPF))
    
    print('nsga',funcName,np.mean(gdNSGA))
    print('spea',funcName,np.mean(gdSPEA))
    print('moga',funcName,np.mean(gdMOGA))
    print('cego',funcName,np.mean(gdCEGO))

problem = Problem(7,2,11)
problem.types[:] = [Real(2.6,3.6),Real(0.7,0.8),Real(17,28),Real(7.3,8.3),Real(7.3,8.3),Real(2.9,3.9),Real(5,5.5)]
problem.constraints[:] = "<=0"
problem.function = SRD
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'SRD'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([7000,1700])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)

problem = Problem(3,2,3)
problem.types[:] = [Real(1,3),Real(0.0005,0.05),Real(0.0005,0.05)]
problem.constraints[:] = "<=0"
problem.function = TBTD
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'TBTD'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([0.1,100000])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(4,2,5)
problem.types[:] = [Real(0.125,5),Real(0.1,10),Real(0.1,10),Real(0.125,5)]
problem.constraints[:] = "<=0"
problem.function = WB
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'WB'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([350,0.1])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(4,2,5)
problem.types[:] = [Real(55,80),Real(75,110),Real(1000,3000),Real(2,20)]
problem.constraints[:] = "<=0"
problem.function = DBD
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'DBD'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([5,50])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(6,3,9)
problem.types[:] = [Real(150,274.32),Real(25,32.31),Real(12,22),Real(8,11.71),Real(14,18),Real(0.63,0.75)]
problem.constraints[:] = "<=0"
problem.function = SPD
algorithm = NSGAII(problem,200)
algorithm.run(200000)
funcname = 'SPD'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([16,19000,-260000])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(7,3,10)
problem.types[:] = [Real(0.5,1.5),Real(0.45,1.35),Real(0.5,1.5),Real(0.5,1.5),Real(0.875,2.625),Real(0.4,1.2),Real(0.4,1.2)]
problem.constraints[:] = "<=0"
problem.function = CSI
algorithm = NSGAII(problem,200)
algorithm.run(200000)
funcname = 'CSI'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([42,4.5,13])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(3,5,7)
problem.types[:] = [Real(0.01,0.45),Real(0.01,0.1),Real(0.01,0.1)]
problem.constraints[:] = "<=0"
problem.function = WP
algorithm = NSGAII(problem,200)
algorithm.run(300000)
funcname = 'WP'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([83000, 1350, 2.85, 15989825, 25000])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(2,2,2)
problem.types[:] = [Real(0,5),Real(0,3)]
problem.constraints[:] = "<=0"
problem.function = BNH
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'BNH'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([140,50])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(2,2,2)
problem.types[:] = [Real(0.1,1),Real(0,5)]
problem.constraints[:] = "<=0"
problem.function = CEXP
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'CEXP'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([1,9])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(6,2,2)
problem.types[:] = [Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1)]
problem.constraints[:] = "<=0"
problem.function = C3DTLZ4
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'C3DTLZ4'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([3,3])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(2,2,2)
problem.types[:] = [Real(-20,20),Real(-20,20)]
problem.constraints[:] = "<=0"
problem.function = SRN
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'SRN'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([301,72])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(2,2,2)
problem.types[:] = [Real(1e-5,np.pi),Real(1e-5,np.pi)]
problem.constraints[:] = "<=0"
problem.function = TNK
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'TNK'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([3,3])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)


problem = Problem(6,2,6)
problem.types[:] = [Real(0,10),Real(0,10),Real(1,5),Real(0,6),Real(1,5),Real(0,10)]
problem.constraints[:] = "<=0"
problem.function = OSY
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'OSY'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([0,386])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurementsOSY(refPF, funcname, ref)


problem = Problem(2,2,2)
problem.types[:] = [Real(0,1),Real(0,1)]
problem.constraints[:] = "<=0"
problem.function = CTP1
algorithm = NSGAII(problem,200)
algorithm.run(100000)
funcname = 'CTP1'    
nondominated_solutions = nondominated(algorithm.result)
ref = np.array([1,2])
refPF = []
for s in nondominated_solutions:
    lijst = str(s.objectives)
    refPF.append(ast.literal_eval(lijst))
refPF = np.array(refPF)
doMeasurement(refPF, funcname, ref)
