# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 14:36:35 2017

@author: r.dewinter
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def visualizeParetoFront3d(results):
    fig = plt.figure()
    plt.title('Pareto Frontier')
    ax = fig.add_subplot(111,projection='3d')
    ax.scatter(results[:,0], results[:,1], results[:,2])
    plt.show()
    
def visualizeParetoFront3dSave(results, outdir=None, objNames=None):
    fig = plt.figure()
#    plt.title('Pareto Frontier')
    ax = fig.add_subplot(111,projection='3d')
    ax.text2D(0.05, 0.95, "Pareto Frontier", transform=ax.transAxes)
    if len(objNames)==3:
        ax.set_xlabel(objNames[0])
        ax.set_ylabel(objNames[1])
        ax.set_zlabel(objNames[2])
    else:
        ax.set_xlabel('dim 1')
        ax.set_ylabel('dim 2')
        ax.set_zlabel('dim 3')
    ax.scatter(results[:,0], results[:,1], results[:,2])
    if outdir is not None:
        plt.savefig(str(outdir)+'ParetoFrontier.pdf',dpi=600)
    else:
        plt.savefig('ParetoFrontier.pdf',dpi=600)
    plt.show()
    
    
    
def visualizeParetoFront2d(results):
    plt.title('Pareto Frontier')
    plt.plot(results[:,0], results[:,1], 'ro')
    plt.show()
    
def visualizeParetoFront2dSave(results, outdir=None, objNames=None):
    plt.title('Pareto Frontier')
    plt.plot(results[:,0], results[:,1], 'ro')
    if len(objNames)==2:
        plt.xlabel(objNames[0])
        plt.ylabel(objNames[1])
    else:
        plt.xlabel('dim 1')
        plt.ylabel('dim 2')
    if outdir is not None:
        plt.savefig(str(outdir)+'ParetoFrontier.pdf',dpi=600)
    else:
        plt.savefig('ParetoFrontier.pdf',dpi=600)
    plt.show()



def visualiseParetoFront(results, save=False, outdir=None, objNames=None):
    if save:
        if results.shape[1]==3:
            visualizeParetoFront3dSave(results, outdir, objNames)
        elif results.shape[1]==2:
            visualizeParetoFront2dSave(results, outdir, objNames)
        else:
            print("To many dimensions to save PF")
    else:
        if results.shape[1]==3:
            visualizeParetoFront3d(results)
        elif results.shape[1]==2:
            visualizeParetoFront2d(results)
        else:
            print("To many dimensions to show PF")