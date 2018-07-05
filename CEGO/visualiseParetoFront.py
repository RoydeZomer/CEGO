# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:36:35 2017

@author: r.dewinter
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualizeParetoFront3d(results):
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(results[:,0], results[:,1], results[:,2])
    plt.show()
    
def visualizeParetoFront2d(results):
    plt.plot(results[:,0], results[:,1], 'ro')
    plt.show()

def visualiseParetoFront(results):
    if results.shape[1]==3:
        visualizeParetoFront3d(results)
    elif results.shape[1]==2:
        visualizeParetoFront2d(results)
    else:
        print("To many dimensions to show PF so far")