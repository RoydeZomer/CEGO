# -*- coding: utf-8 -*-
"""
Created on Mon May 14 21:47:57 2018

@author: r.dewinter
"""

import numpy as np
import matplotlib.pyplot as plt

ax = plt.subplot(111)  
ax.plot([2], [2],'k*',label="Reference Point")
ax.plot([1.8,1.5,1], [1,1.5,1.8],'b^',label="Non Dominant Solutions")
ax.fill_between(x=[1.2,2],y1=[1.2,1.2],y2=2,alpha=0.5,label="S-Metric Selection Score",color='g')
ax.plot([2,1.7,1.7,1.4,1.4,0.9,0.9],[0.9,0.9,1.4,1.4,1.7,1.7,2], label=r'$\epsilon$ area',color='gray')
ax.fill_between(x=[2,1.8,1.8,1.5,1.5,1,1],y1=[1,1,1.5,1.5,1.8,1.8,2],y2=2,alpha=1,label='Current Hypervolume',color='lightgray')
ax.plot([1.65], [1.65],'o',color='r',label=r'Dominated Solution')
ax.plot([1.75], [1.05],'o',color='darkgoldenrod',label=r'$\epsilon$-Dominated Solution')
ax.plot([1.2], [1.2],'go',label=r'Non-$\epsilon$-Dominated Solution')

ax.set_xlabel('Objective 1')
ax.set_ylabel('Objective 2')
ax.set_title('S-Metric Selection infill criterion')

ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.legend(bbox_to_anchor=(1, 0.5), loc=2, borderaxespad=0.,framealpha=0.)
plt.savefig("smsego.PDF",dpi=600,bbox_inches='tight') 

plt.show()

