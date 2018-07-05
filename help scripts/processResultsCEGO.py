# -*- coding: utf-8 -*-
"""
Created on Thu Mar 22 18:25:11 2018

@author: r.dewinter
"""

from hypervolume import hypervolume
import numpy as np

hyp = []
runs = ['run4','run7','run8','run9','run14']
funcname = 'SRD'
ref = np.array([7000,1700])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))

hyp = []
runs = ['run1','run2','run3','run4','run5']
funcname = 'TBTD'
ref = np.array([0.1,100000])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))

hyp = []
runs = ['run1','run2','run3','run4','run5']
funcname = 'WB'
ref = np.array([350,0.1])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))

hyp = []
runs = ['run1','run2','run3','run4','run5']
funcname = 'DBD'
ref = np.array([5,50])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))

hyp = []
runs = ['run1','run4','run5','run6','run7']
funcname = 'SPD'
ref = np.array([16,19000,-260000])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))

hyp = []
runs = ['run1','run2','run3','run4','run5']
funcname = 'WP'
ref = np.array([83000, 1350, 2.85, 15989825, 25000])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))

hyp = []
runs = ['run4','run7','run6','run10','run11']
funcname = 'CSI'
ref = np.array([42,4.5,13])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))


############################################# artificial 

hyp = []
runs = ['run1','run2','run3','run4','run10']
funcname = 'BNH'
ref = np.array([140,50])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))


hyp = []
runs = ['run1','run2','run3','run5','run6']
funcname = 'CEXP'
ref = np.array([1,9])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))


hyp = []
runs = ['run1','run9','run10','run7','run8']
funcname = 'C3DTLZ4'
ref = np.array([3,3])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))


hyp = []
runs = ['run1','run2','run3','run4','run5']
funcname = 'SRN'
ref = np.array([301,72])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))


hyp = []
runs = ['run1','run2','run3','run4','run10']
funcname = 'TNK'
ref = np.array([3,3])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))

hyp = []
runs = ['run2','run4','run5','run6','run7']
funcname = 'OSY'
ref = np.array([0,386])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))


hyp = []
runs = ['run1','run2','run3','run4','run5']
funcname = 'CTP1'
ref = np.array([1,2])
for run in runs:
    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
    objectives = np.genfromtxt(file, delimiter=',')
    hyp.append(hypervolume(objectives,ref))
print(funcname,np.mean(hyp))
print(funcname,np.max(hyp))
print(funcname,np.std(hyp))

#hyp = []
#runs = ['run3']
#funcname = 'DTLZ8'
#ref = np.array([1,1,1])
#for run in runs:
#    file = funcname+"/"+run+"/obj_"+run+"_finalPF.csv"
#    objectives = np.genfromtxt(file, delimiter=',')
#    hyp.append(hypervolume(objectives,ref))
#print(funcname,np.mean(hyp))
#print(funcname,np.max(hyp))
#print(funcname,np.std(hyp))
