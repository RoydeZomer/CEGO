## Overview

**CEGO** *(Constrained Multi-Objective Efficient Global Optimization)* is an optimization algorithm that can be used to optimize constrained multi-objective optimization problems using a limited number of function evaluations. 

The advantage of CEGO is that it uses a surrogate models for both the constraints and the objectives to learn from the evaluations it has made so far. For the constraints Cubic Radial Basis Functions are used and for the objectives we use Gaussian Process Regression in combination with the S-Metric Selection criterion. 

For example, one use case would be to optimize expensive objectives that need to be computed with simulations. See the following paper (yet to be published) and master thesis for more detail about this algorithm:  
Winter, et al. *Designing Ships using Constrained Multi-Objective Efficient Global Optimization.* https://dx.doi.org/10.13140/RG.2.2.21395.12328

## Pseudo code:

![alt text](https://github.com/RoydeZomer/CEGO/pseudocode.PNG)

## Usage

To use the optimization algorithm you need to define an objective function, the constraint function, and the search space before you can start the optimizer. Below is an examples that describe most of the functionality.

### Example - Optimizing CEXP problem

```python
import numpy as np

#imports from our package
from CONSTRAINED_SMSEGO import CONSTRAINED_SMSEGO


# The "black-box" function
# returns objective array, and constraint array
# objective is to be minimized
# constraint should be transformed so that values samller then or equal to 0 are feasible
def CEXP(x):
    x1 = x[0]
    x2 = x[1]
    
    f1 = x1
    f2 = (1+x2)/x1
    
    g1 = x2 + 9*x1 - 6
    g2 = -1*x2 + 9*x1 - 1
    
    objectives = np.array([f1, f2])
    constraints = np.array([g1,g2])
    constraints = -1*constraints 
    return objectives, constraints

problemCall = CEXP
nconstraints = 2

# First we need to define the Search Space
# the search space consists of two continues variables in [0.1,1] and [0,5]
rngMin = np.array([0.1,0])
rngMax = np.array([1,5])

#the objective space boundary is simply the largest values of the objective function we are interested in.
ref = np.array([1,9])

#and we run the optimization.
obj, constr, param = CONSTRAINED_SMSEGO(problemCall, rngMin, rngMax, ref, nconstraints)

```

Other examples can be found in constraintsmsegocall.py


If you are interested in only one CEGO iteration you can use the CEGOIteration function.

```python
import numpy as np

#imports from our package
from CEGOIteration import CEGOIteration

problemCall = CEXP
nconstraints = 2

# First we need to define the Search Space
# the search space consists of two continues variables in [0.1,1] and [0,5]
rngMin = np.array([0.1,0])
rngMax = np.array([1,5])

#the objective space boundary is simply the largest values of the objective function we are interested in.
ref = np.array([1,9])

#create initial sample
param = np.random.uniform(low=rngMin, high=rngMax, size=(50,2))
constr = np.empty((50,2))
obj = np.empty((50,2))
for i in range(len(param)):
    obj[i], constr[i] = problemCall(param[i])
dat = np.column_stack((param,constr,obj))

#and we run 1 iteration.
obj, constr, param = CEGOIteration(problemCall, rngMin, rngMax, ref, nconstraints, data=dat)

```
