# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 10:36:45 2017

@author: r.dewinter
"""
import numpy as np

def paretofrontFeasible(costs, constraints):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    
    n_constraints = constraints.shape[1]
    feasible = sum(constraints <= 0, axis = 1) == n_constraints
    
    indexes = np.arange(len(feasible))
    findexes = indexes[feasible]
    n_points1 = costs.shape[0]
    
    costs = costs[findexes]
    
    is_efficient = np.arange(costs.shape[0])
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<=costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1

    pff = findexes[is_efficient]
    
    is_efficient_mask = np.zeros(n_points1, dtype = bool)
    is_efficient_mask[pff] = True
    
    return is_efficient_mask
