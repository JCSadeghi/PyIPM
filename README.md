
# PyIPM
## Jonathan Sadeghi, University of Liverpool, 2018

[![Build Status](https://travis-ci.org/JCSadeghi/PyIPM.svg?branch=master)](https://travis-ci.org/JCSadeghi/PyIPM)

This code is a port of the MATLAB Interval Predictor Model class from the OpenCossan generalised uncertainty quantification software. The code was tested in Python 2.7.
This version of the code is significantly simplified. If the model obtained is too conservative for your requirements then you may wish to download [OpenCossan](http://www.cossan.co.uk), which is freely available after registration. The MATLAB version of the code contains many optimisations to improve the performance of the models.

[Interval Predictor Model Code](http://personalpages.manchester.ac.uk/staff/jonathan.sadeghi/codes.htm)

If you find the toolbox useful for your research and decide to use it in a paper we kindly request that you cite the following paper:

[E Patelli, M Broggi, S Tolo, J Sadeghi, Cossan Software: A Multidisciplinary And Collaborative Software For Uncertainty Quantification, UNCECOMP 2017, At Rhodes Island, Greece, 2nd ECCOMAS Thematic Conference on Uncertainty Quantification in Computational Sciences and Engineering, June 2017.](https://www.researchgate.net/publication/317598944_COSSAN_SOFTWARE_A_MULTIDISCIPLINARY_AND_COLLABORATIVE_SOFTWARE_FOR_UNCERTAINTY_QUANTIFICATION)

The code requires the cvxopt package (pip install cvxopt), sklearn and numpy.

Here is a brief demonstration of the code:

First import the required dependancies:

```python
import numpy as np
from PyIPM import PyIPM
```

Construct an Interval Predictor Model:

```python
model=PyIPM(polynomialDegree=2)
```

Sample 100 realisations from an arbitrary function to emulate training data. Two input features and one output:
```python
x=5*(np.random.rand(100,2)-0.5)
y=x[:,0]**2+x[:,1]**2*np.random.rand(1,100)
```

Fit the model to the training data:
```python
model.fit(x,y[0,:])
```

Get upper and lower bound of model on the training data set:
```python
upperBound,lowerBound=model.predict(x)
```

Plot the training data with the model predictions:
```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], y,color='black')
ax.scatter(x[:,0], x[:,1], lowerBound[:,0])
ax.scatter(x[:,0], x[:,1], upperBound[:,0])
fig
```

![png](output_6_0.png)


```python
model.getModelReliability()
```

0.8217819999998218
