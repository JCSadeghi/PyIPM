import numpy as np
from PyIPM import PyIPM

np.random.seed(0)

model=PyIPM(polynomialDegree=2)

x=5*(np.random.rand(100,2)-0.5)
y=x[:,0]**2+x[:,1]**2*np.random.rand(1,100)

model.fit(x,y[0,:])

upperBound,lowerBound=model.predict(x)

tolerance=0.01

assert(upperBound[0]-1.15<tolerance), "Upper bound does not match stored value"

assert(lowerBound[0]-0.15<tolerance), "Lower bound does not match stored value"

ModelReliability=model.getModelReliability()

assert(ModelReliability-0.88<tolerance), "Model reliability does not match stored value"
