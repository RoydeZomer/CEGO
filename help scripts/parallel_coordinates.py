# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 09:13:35 2018

@author: r.dewinter
"""
import numpy as np
import matplotlib.pyplot as plt
from pandas.tools.plotting import parallel_coordinates
from paretofrontFeasible import paretofrontFeasible

import matplotlib.colors as color
import matplotlib.ticker as ticker

import mpld3

def parallel_coordinates(data_sets, colors=None, columNames=None, alpha=None):
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
#    fig.savefig("paralelcoordinate1.pdf",dpi=600,bbox_inches='tight') 
    #fig.savefig("paralelcoordinate")
    mpld3.display(fig)

constraints = np.genfromtxt('P:/17.xxx Projecten 2017/17.502 Internship Roy de Winter/code/CONSTRAINED_EGOresults/compute_ship/run300/con_run300.csv', delimiter=',')
parameters = np.genfromtxt('P:/17.xxx Projecten 2017/17.502 Internship Roy de Winter/code/CONSTRAINED_EGOresults/compute_ship/run300/par_run300.csv', delimiter=',')
objectives = np.genfromtxt('P:/17.xxx Projecten 2017/17.502 Internship Roy de Winter/code/CONSTRAINED_EGOresults/compute_ship/run300/obj_run300.csv', delimiter=',')

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
data = data[:,1:] #remove rank
alpha = alpha[order]
sort = np.argsort([0,2,3,1,4,5])
data = data[:,sort]
columNames=['HOPPER EXTENSION','HOPPER WIDTH','HOPPER HEIGHT','FORESHIP LENGTH','SHIP LENGTH','SHIP BREADTH']

parallel_coordinates(data, colors, columNames, alpha)
#
#idx = rank==2
#data = data[idx]
#colors = colors[idx]
#alpha = alpha[idx]
#
#parallel_coordinates(data, colors, columNames, alpha)
