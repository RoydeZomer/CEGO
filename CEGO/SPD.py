# -*- coding: utf-8 -*-
"""
Created on Mon Mar  5 12:11:03 2018

@author: r.dewinter
"""

import numpy as np

#def write_results(constraints, objectives):
#    try:
#        con = np.genfromtxt('conSPD.csv',delimiter=',')
#        obj = np.genfromtxt('objSPD.csv',delimiter=',')
#        con = np.asmatrix(con)
#        obj = np.asmatrix(obj)
#    except:
#        con = np.empty((0,9))
#        obj = np.empty((0,3))
#    con = np.append(con,constraints,axis=0)
#    obj = np.append(obj,objectives,axis=0)
#    np.savetxt('conSPD.csv',con,delimiter=',')
#    np.savetxt('objSPD.csv',obj,delimiter=',')


def SPD(x):
    L = x[0]
    B = x[1]
    D = x[2]
    T = x[3]
    Vk = x[4]
    Cb = x[5]

    displacement = 1.025*L*B*T*Cb
    V = 0.5144*Vk
    g = 9.8065
    Fn = V/((g*L)**0.5)
    b = -10847.2*(Cb**2)+12817*Cb - 6960.32
    a = 4977.06*(Cb**2) - 8105*Cb + 4456.51
    P = (displacement**(2/3))*(Vk**3)/(a+b*Fn)
    Wm = 0.17*(P**0.9)
    Wo = 1*(L**0.8)*(B**0.6)*(D**0.3)*(Cb**0.1)
    Ws = 0.034*(L**1.7)*(B**0.7)*(D**0.4)*(Cb**0.5)
    lightShip = Ws+Wo+Wm    
    KG = 1+0.52*D
    BMt = (0.085*Cb-0.002)*(B**2)/(T*Cb)
    KB = 0.53*T
    DWT = displacement - lightShip
    handlingRate = 8000
    miscellaneousDWT = 2*(DWT**0.5)
    dailyConsumption = 0.19*P*24/1000+0.2
    roundTripMiles = 5000
    seaDays = roundTripMiles/(24*Vk)
    fuelCarried = dailyConsumption*(seaDays+5)
    cargoDeadweight = DWT - fuelCarried-miscellaneousDWT
    portDays = 2*((cargoDeadweight/handlingRate)+0.5)
    RTPA = 350/(seaDays+portDays)
    portCost = 6.3*(DWT**0.8)
    fuelPrice = 100
    runningCosts = 40000*(DWT**0.3)
    fuelCost = 1.05*dailyConsumption*seaDays*fuelPrice
    voyageCosts = (fuelCost+portCost)*RTPA
    shipCost = 1.3*(2000*Ws**0.85+3500*Wo+2400*P**0.8)
    capitalCosts = 0.2*shipCost
    annualCost = capitalCosts + runningCosts+voyageCosts
    
    annualCargo = cargoDeadweight * RTPA
    transportationCost = annualCost/annualCargo
    annualCargo = annualCargo * -1 
    
    g1 = -1*(L/B - 6)
    g2 = L/D - 15
    g3 = L/T - 19
    g4 = T - 0.45*(DWT**(0.31))
    g5 = T - 0.7*D - 0.7
    g6 = 3000 - DWT
    g7 = DWT-500000
    g8 = Fn - 0.32
    g9 = -1*(KB+BMt-KG - 0.07*B)
    
#    write_results([np.array([g1,g2,g3,g4,g5,g6,g7,g8,g9])],[np.array([transportationCost, lightShip, annualCargo])])
    
    return np.array([transportationCost, lightShip, annualCargo]), np.array([g1,g2,g3,g4,g5,g6,g7,g8,g9])

#import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D
#from paretofrontFeasible import paretofrontFeasible
#rngMin = np.array([150,    25,    12,   8,     14, 0.63])
#rngMax = np.array([274.32, 32.31, 22,   11.71, 18, 0.75])
#nVar = 6
#ref = np.array([16,19000,-260000])
#parameters = np.empty((1000000,6))
#objectives = np.empty((1000000,3))
#constraints = np.empty((1000000,9))
#objectives[:] = 0
#constraints[:] = 0
#parameters[:]= 0
#for i in range(1000000):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    parameters[i] = x
#    obj, cons = SPD(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#a = np.sum(constraints<0, axis=1)==9
##sum(a)
#fig = plt.figure()
#ax = fig.add_subplot(111,projection='3d')
##ax.scatter(objectives[a][:,0], objectives[a][:,1], objectives[a][:,2])
#b = paretofrontFeasible(objectives[a],np.zeros(objectives[a].shape))
#ax.scatter(objectives[a][b][:,0], objectives[a][b][:,1], objectives[a][b][:,2])
#fig.show()
    
#iteration time 175.67300009727478
#53445.14100027084
#53498.00699996948
    
#iteration time 173.2409999370575
#17932.417999982834
#17985.308000087738