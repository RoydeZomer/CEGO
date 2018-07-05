# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 13:37:34 2018

@author: r.dewinter
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 11:59:44 2018

@author: r.dewinter
"""
from TBTD import TBTD
from SRD import SRD
from WB import WB
from DBD import DBD
from SPD import SPD
from CSI import CSI
from WP import WP

from DTLZ8 import DTLZ8
from OSY import OSY
from CTP1 import CTP1
from CEXP import CEXP
from C3DTLZ4 import C3DTLZ4
from TNK import TNK
from SRN import SRN
from BNH import BNH

import numpy as np
from platypus import SPEA2
from platypus import Problem
from platypus import Real
from platypus import nondominated
from hypervolume import hypervolume
import ast
import os 

########################## RWLP
#SRD 1463569.1268
#SRD 3455378.79582
#SRD 839120.546161
#
#TBTD 7064.14408981
#TBTD 8195.97127377
#TBTD 674.284772683
#
#WB 33.7159848971
#WB 34.4366080032
#WB 0.342053328936
#
#DBD 214.499085504
#DBD 220.527506569
#DBD 3.23787180398
#
#SPD 20094386654.8
#SPD 27212325016.4
#SPD 4575705423.85
#
#CSI 13.9836855119
#CSI 16.0148038568
#CSI 1.17440237345
#
#WP 1.126607818e+19
#WP 1.24357860884e+19
#WP 6.50457124301e+17
#

################################# theoretics
#BNH 5135.23509014
#BNH 5175.75512319
#BNH 20.4229406427
#
#CEXP 3.15841637789
#CEXP 3.4304763721
#CEXP 0.149550919484
#
#DTLZ8 0.512261429031
#DTLZ8 0.563039150214
#DTLZ8 0.0242657708823
#
#C3DTLZ4 5.04753543194
#C3DTLZ4 5.59183053986
#C3DTLZ4 0.175165766953
#
#SRN 56293.0183148
#SRN 62532.5869024
#SRN 3277.86775054
#
#TNK 6.44822171504
#TNK 7.41954845288
#TNK 0.514295614025
#
#OSY 21672.4112602
#OSY 76708.9006817
#OSY 15495.3973593
#
#CTP1 1.220802764
#CTP1 1.24770231121
#CTP1 0.0163707270088

hyp = []
for i in range(5):
    problem = Problem(6,2,16)
    problem.types[:] = [Real(5.0,9.0),Real(16.0,22.0),Real(5.0,9.0),Real(12.0,16.0),Real(-2.8,9.8),Real(-1.6,3.4)]
    problem.constraints[:] = "<=0"
    problem.function = problemCall
    algorithm = SPEA2(problem)
    algorithm.run(200)
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([5000,2])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    hyp.append(hypervolume(obj,ref))
print('SRD',np.mean(hyp))
print('SRD',np.max(hyp))
print('SRD',np.std(hyp))


hyp = []
for i in range(100):
    problem = Problem(7,2,11)
    problem.types[:] = [Real(2.6,3.6),Real(0.7,0.8),Real(17,28),Real(7.3,8.3),Real(7.3,8.3),Real(2.9,3.9),Real(5,5.5)]
    problem.constraints[:] = "<=0"
    problem.function = SRD
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    funcname = 'SRD'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([7000,1700])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    
    hyp.append(hypervolume(obj,ref))
print('SRD',np.mean(hyp))
print('SRD',np.max(hyp))
print('SRD',np.std(hyp))


hyp = []
for i in range(100):
    problem = Problem(3,2,3)
    problem.types[:] = [Real(1,3),Real(0.0005,0.05),Real(0.0005,0.05)]
    problem.constraints[:] = "<=0"
    problem.function = TBTD
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    funcname = 'TBTD'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([0.1,100000])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('TBTD',np.mean(hyp))
print('TBTD',np.max(hyp))
print('TBTD',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(4,2,5)
    problem.types[:] = [Real(0.125,5),Real(0.1,10),Real(0.1,10),Real(0.125,5)]
    problem.constraints[:] = "<=0"
    problem.function = WB
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    funcname = 'WB'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([350,0.1])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('WB',np.mean(hyp))
print('WB',np.max(hyp))
print('WB',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(4,2,5)
    problem.types[:] = [Real(55,80),Real(75,110),Real(1000,3000),Real(2,20)]
    problem.constraints[:] = "<=0"
    problem.function = DBD
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    
    funcname = 'DBD'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([5,50])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('DBD',np.mean(hyp))
print('DBD',np.max(hyp))
print('DBD',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(6,3,9)
    problem.types[:] = [Real(150,274.32),Real(25,32.31),Real(12,22),Real(8,11.71),Real(14,18),Real(0.63,0.75)]
    problem.constraints[:] = "<=0"
    problem.function = SPD
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    
    funcname = 'SPD'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([16,19000,-260000])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('SPD',np.mean(hyp))
print('SPD',np.max(hyp))
print('SPD',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(7,3,10)
    problem.types[:] = [Real(0.5,1.5),Real(0.45,1.35),Real(0.5,1.5),Real(0.5,1.5),Real(0.875,2.625),Real(0.4,1.2),Real(0.4,1.2)]
    problem.constraints[:] = "<=0"
    problem.function = CSI
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    
    funcname = 'CSI'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([42,4.5,13])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('CSI',np.mean(hyp))
print('CSI',np.max(hyp))
print('CSI',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(3,5,7)
    problem.types[:] = [Real(0.01,0.45),Real(0.01,0.1),Real(0.01,0.1)]
    problem.constraints[:] = "<=0"
    problem.function = WP
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    funcname = 'WP'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([83000, 1350, 2.85, 15989825, 25000])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('WP',np.mean(hyp))
print('WP',np.max(hyp))
print('WP',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(2,2,2)
    problem.types[:] = [Real(0,5),Real(0,3)]
    problem.constraints[:] = "<=0"
    problem.function = BNH
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    
    funcname = 'BNH'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([140,50])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('BNH',np.mean(hyp))
print('BNH',np.max(hyp))
print('BNH',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(2,2,2)
    problem.types[:] = [Real(0.1,1),Real(0,5)]
    problem.constraints[:] = "<=0"
    problem.function = CEXP
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    
    funcname = 'CEXP'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([1,9])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('CEXP',np.mean(hyp))
print('CEXP',np.max(hyp))
print('CEXP',np.std(hyp))

#hyp = []
#for i in range(100):
#    problem = Problem(9,3,3)
#    problem.types[:] = [Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1)]
#    problem.constraints[:] = "<=0"
#    problem.function = DTLZ8
#    algorithm = SPEA2(problem)
#    algorithm.run(200)
#    
#    funcname = 'DTLZ8'
#    if not os.path.exists(funcname):
#        os.makedirs(funcname)
#    objectives = np.array([[obj for obj in s.objectives] for s in algorithm.result])
#    parameters = np.array([[obj for obj in s.variables] for s in algorithm.result])
#    constraints = np.array([[obj for obj in s.constraints] for s in algorithm.result])
#    np.savetxt(str(funcname)+'/'+str(funcname)+'_parameters_run_'+str(i)+'.csv', parameters, delimiter=',')
#    np.savetxt(str(funcname)+'/'+str(funcname)+'_objectives_run_'+str(i)+'.csv', objectives, delimiter=',')
#    np.savetxt(str(funcname)+'/'+str(funcname)+'_constraints_run_'+str(i)+'.csv', constraints, delimiter=',')
#    
#    
#    nondominated_solutions = nondominated(algorithm.result)
#    ref = np.array([1,1,1])
#    obj = []
#    for s in nondominated_solutions:
#        lijst = str(s.objectives)
#        obj.append(ast.literal_eval(lijst))
#    obj = np.array(obj)
#    hyp.append(hypervolume(obj,ref))
#print('DTLZ8',np.mean(hyp))
#print('DTLZ8',np.max(hyp))
#print('DTLZ8',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(6,2,2)
    problem.types[:] = [Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1),Real(0,1)]
    problem.constraints[:] = "<=0"
    problem.function = C3DTLZ4
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    funcname = 'C3DTLZ4'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([3,3])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('C3DTLZ4',np.mean(hyp))
print('C3DTLZ4',np.max(hyp))
print('C3DTLZ4',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(2,2,2)
    problem.types[:] = [Real(-20,20),Real(-20,20)]
    problem.constraints[:] = "<=0"
    problem.function = SRN
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    
    funcname = 'SRN'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([301,72])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('SRN',np.mean(hyp))
print('SRN',np.max(hyp))
print('SRN',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(2,2,2)
    problem.types[:] = [Real(1e-5,np.pi),Real(1e-5,np.pi)]
    problem.constraints[:] = "<=0"
    problem.function = TNK
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    funcname = 'TNK'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([3,3])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('TNK',np.mean(hyp))
print('TNK',np.max(hyp))
print('TNK',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(6,2,6)
    problem.types[:] = [Real(0,10),Real(0,10),Real(1,5),Real(0,6),Real(1,5),Real(0,10)]
    problem.constraints[:] = "<=0"
    problem.function = OSY
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    funcname = 'OSY'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([0,386])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('OSY',np.mean(hyp))
print('OSY',np.max(hyp))
print('OSY',np.std(hyp))

hyp = []
for i in range(100):
    problem = Problem(2,2,2)
    problem.types[:] = [Real(0,1),Real(0,1)]
    problem.constraints[:] = "<=0"
    problem.function = CTP1
    algorithm = SPEA2(problem)
    algorithm.run(200)
    
    funcname = 'CTP1'
    if not os.path.exists(funcname):
        os.makedirs(funcname)
        
    nondominated_solutions = nondominated(algorithm.result)
    ref = np.array([1,2])
    obj = []
    for s in nondominated_solutions:
        lijst = str(s.objectives)
        obj.append(ast.literal_eval(lijst))
    obj = np.array(obj)
    np.savetxt(str(funcname)+'/'+str(funcname)+'_pf_run_'+str(i)+'.csv', obj, delimiter=',')
    hyp.append(hypervolume(obj,ref))
print('CTP1',np.mean(hyp))
print('CTP1',np.max(hyp))
print('CTP1',np.std(hyp))