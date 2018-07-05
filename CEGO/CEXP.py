# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 10:38:07 2018

@author: r.dewinter
"""
import numpy as np 

#def write_results(constraints, objectives):
#    try:
#        con = np.genfromtxt('conCEXP.csv',delimiter=',')
#        obj = np.genfromtxt('objCEXP.csv',delimiter=',')
#        con = np.asmatrix(con)
#        obj = np.asmatrix(obj)
#    except:
#        con = np.empty((0,2))
#        obj = np.empty((0,2))
#    con = np.append(con,constraints,axis=0)
#    obj = np.append(obj,objectives,axis=0)
#    np.savetxt('conCEXP.csv',con,delimiter=',')
#    np.savetxt('objCEXP.csv',obj,delimiter=',')

def CEXP(x):
    x1 = x[0]
    x2 = x[1]
    
    f1 = x1
    f2 = (1+x2)/x1
    
    g1 = x2 + 9*x1 - 6
    g2 = -1*x2 + 9*x1 - 1
    
    objectives = np.array([f1, f2])
    constraints = np.array([g1,g2])
    constraints = -1*constraints #transform for sacobra
#    write_results([constraints],[objectives])
    return np.array([objectives, constraints])


#import matplotlib.pyplot as plt
#rngMin = np.array([0.1,0])
#rngMax = np.array([1,5])
#nVar = 2
#nconstr = 2
#ref = np.array([1,9])
#randomN = 100000
#parameters = np.empty((randomN,nVar))
#objectives = np.empty((randomN,2))
#constraints = np.empty((randomN,nconstr))
#objectives[:] = 0
#constraints[:] = 0
#parameters[:]= 0
#for i in range(randomN):
#    x = np.random.rand(nVar)*(rngMax-rngMin)+rngMin
#    parameters[i] = x
#    obj, cons = CEXP(x)
#    objectives[i] = obj
#    constraints[i] = cons
#
#a = np.sum(constraints<0, axis=1)==nconstr
##sum(a)
#plt.plot(objectives[a][:,0], objectives[a][:,1],'ro')
#plt.plot(objectives[:,0], objectives[:,1],'ro')
#    
#pf = paretofrontFeasible(objectives,constraints)
#obj = objectives[pf]
#objI = np.argsort(obj[:,0],axis=0)
#plt.plot(obj[objI][:,0],obj[objI][:,1])
#plt.plot(ref[0],ref[1],'ro')
#
#ax = plt.subplot(111)
#ax.plot(obj[objI][:,0],obj[objI][:,1],c='g',linewidth=5.0, label='Pareto Frontier')
#ax.plot(ref[0],ref[1],'o',c='r', label='Reference point')
#ax.fill_between(x=obj[objI][:,0],y1=obj[objI][:,1],y2=9,alpha=0.5,label='(hyper)volume')
#ax.set_xlabel('f1')
#ax.set_ylabel('f2')
#ax.set_title('CEXP')
#ax.spines['right'].set_visible(False)
#ax.spines['top'].set_visible(False)
#plt.legend(bbox_to_anchor=(0.6, 0.6), loc=2, borderaxespad=0.1,framealpha=1.)
#plt.savefig("CEXPPFREF.pdf",dpi=600,bbox_inches='tight') 

#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.pyplot as plt
#from matplotlib import cm
#from matplotlib.ticker import LinearLocator, FormatStrFormatter
#
#for p in range(constraints.shape[1]):
#    x = parameters[:,0]
#    x2 = parameters[:,1]
#    z = constraints[:,p]
#        
#    fig = plt.figure()
#    ax = fig.gca(projection='3d')
#        
#    ax.plot_trisurf(x, x2, z, linewidth=0.2, antialiased=True)


#iteration time 102.84171462059021
#8881.99292421341
#8891.746877908707
    
#iteration time 25.113999843597412
#2891.483999967575
#2915.2659997940063
    
#iteration time 22.413000345230103
#2825.91300034523
#2848.755999803543