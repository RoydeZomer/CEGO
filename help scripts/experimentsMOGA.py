# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 09:00:23 2018

@author: r.dewinter
"""

import numpy as np
import matplotlib.pyplot as plt

from hypervolume import hypervolume
from paretofrontFeasible import paretofrontFeasible

import os

#plt.plot(objectivesMOGA[:,0],objectivesMOGA[:,1],'ro',c='r')
#plt.plot(objectivesSPEA2[:,0],objectivesSPEA2[:,1],'ro',c='b')
#plt.plot(objectivesNSGAII[:,0],objectivesNSGAII[:,1],'ro',c='g')
#plt.plot(objectivesCEGO[:,0],objectivesCEGO[:,1],'ro',c='m')


hyp = []
fname = 'optimize ship'
ref = np.array([5000,2])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,9:-3]
    constraints[:,:4] = constraints[:,:4]*-1+1
    constraints[:,4:] = constraints[:,4:]*-1
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))


hyp = []
fname = 'SRD'
ref = np.array([7000,1700])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,10:-3]
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))

hyp = []
fname = 'TBTD'
ref = np.array([0.1,100000])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,6:-3]
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))

hyp = []
fname = 'WB'
ref = np.array([350,0.1])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,7:-3]
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))


hyp = []
fname = 'DBD'
ref = np.array([5,50])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,7:-3]
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))

hyp = []
fname = 'SPD'
ref = np.array([16,19000,-260000])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,9:-4]
    objectives = data[1:,-4:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))

hyp = []
fname = 'WP'
ref = np.array([83000, 1350, 2.85, 15989825, 25000])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,6:13]
    objectives = data[1:,13:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))


############################ artificial


hyp = []
fname = 'BNH'
ref = np.array([140,50])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,-5:-3]
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))

hyp = []
fname = 'CEXP'
ref = np.array([1,9])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,-5:-3]
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))


hyp = []
fname = 'C3DTLZ4'
ref = np.array([3,3])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,-5:-3]
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))

hyp = []
fname = 'SRN'
ref = np.array([301,72])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,-5:-3]
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))

hyp = []
fname = 'TNK'
ref = np.array([3,3])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,-5:-3]
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))


hyp = []
fname = 'OSY'
ref = np.array([0,386])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = -1*data[1:,-9:-3] #>0
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))

hyp = []
fname = 'CTP1'
ref = np.array([1,2])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = -1*data[1:,-5:-3] #>0
    objectives = data[1:,-3:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))

hyp = []
fname = 'CSI'
ref = np.array([42,4.5,13])
for file in os.listdir(fname):
    data = np.genfromtxt(fname+'/'+file, delimiter=',')
    constraints = data[1:,-14:-4] 
    objectives = data[1:,-4:-1]
    feasible = np.sum(constraints<=0,axis=1) == constraints.shape[1]
    hyp.append(hypervolume(objectives[feasible],ref))
print(fname,np.mean(hyp))
print(fname,np.max(hyp))
print(fname,np.std(hyp))