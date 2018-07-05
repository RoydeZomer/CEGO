# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:36:45 2017

@author: r.dewinter
"""

import numpy as np

def paretofrontFeasible(objectives, constraints):
    """
    :param objectives: An (n_points, n_objectives) array
           constraints: An (n_points, n_constraints) array
    :return: A (n_points, ) boolean array, indicating whether each point is Pareto efficient,
             pareto efficient points are only the feasible ones
             Feasible: constraints <= 0 for all constraint values. 
    """
    n_constraints = constraints.shape[1]
    feasible = np.sum(constraints <= 0, axis = 1) == n_constraints
    
    S = {} #pareto dominant points
    def update(p):
        if any(all(S[q] <= p) for q in S):
           return
        for q in [q for q in S if all(p<=S[q])]:
            del S[q]
        S[frozenset(p)] = p
    
    for obj in objectives[feasible]:
        update(obj) #add every point to dominant points if it is dominant
    
    S = np.array(list(S.values())) #make list from dominant points
    
    a = (np.sum(np.isin(objectives,S),axis=1)+feasible)==(objectives.shape[1]+1) #make array of booleans to indicate which are dominant
    return(a)