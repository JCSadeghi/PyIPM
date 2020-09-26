# PyIPM
## Jonathan Sadeghi, University of Liverpool, 2018

[![DOI](https://zenodo.org/badge/124306436.svg)](https://zenodo.org/badge/latestdoi/124306436)

This code is a port of the MATLAB Interval Predictor Model class from the OpenCossan generalised uncertainty quantification software. The code was tested in Python 2.7.
This version of the code is significantly simplified. If the model obtained is too conservative for your requirements then you may wish to download [OpenCossan](http://www.cossan.co.uk), which is freely available after registration. The MATLAB version of the code contains many optimisations to improve the performance of the models.

[More information about my work on the Cossan Interval Predictor Model Code](https://jcsadeghi.github.io/blog/2018/math/)

[More information about what this toolbox actually does](https://www.researchgate.net/publication/317598944_COSSAN_SOFTWARE_A_MULTIDISCIPLINARY_AND_COLLABORATIVE_SOFTWARE_FOR_UNCERTAINTY_QUANTIFICATION)

If you find the toolbox useful for your research and decide to use it in a paper we kindly request that you cite the following paper:

[E Patelli, M Broggi, S Tolo, J Sadeghi, Cossan Software: A Multidisciplinary And Collaborative Software For Uncertainty Quantification, UNCECOMP 2017, At Rhodes Island, Greece, 2nd ECCOMAS Thematic Conference on Uncertainty Quantification in Computational Sciences and Engineering, June 2017.](https://www.researchgate.net/publication/317598944_COSSAN_SOFTWARE_A_MULTIDISCIPLINARY_AND_COLLABORATIVE_SOFTWARE_FOR_UNCERTAINTY_QUANTIFICATION)

### Tutorial

Here is a brief demonstration of the code:

First import the required dependencies:



```python
import numpy as np
import PyIPM
```

Construct an Interval Predictor Model:




```python
model = PyIPM.IPM(polynomial_degree=2)
```


Sample 100 realisations from an arbitrary function to emulate training data. Two input features and one output:


```python
x = 5 * (np.random.rand(100, 2) - 0.5)
y = x[:, 0] ** 2 + x[:, 1] ** 2 * np.random.rand(100)
```


Fit the model to the training data:


```python
model.fit(x, y)
```

         pcost       dcost       gap    pres   dres   k/t
     0:  2.2204e-16  6.0908e-15  7e+02  9e-01  1e+02  1e+00
     1:  4.5967e+00  4.5940e+00  4e+01  7e-02  8e+00  8e-02
     2:  3.4183e+00  3.4327e+00  1e+01  3e-02  3e+00  4e-02
     3:  2.5666e+00  2.5764e+00  4e+00  1e-02  1e+00  2e-02
     4:  2.1747e+00  2.1769e+00  7e-01  3e-03  3e-01  5e-03
     5:  2.1922e+00  2.1940e+00  7e-01  3e-03  3e-01  5e-03
     6:  2.1577e+00  2.1585e+00  3e-01  1e-03  1e-01  2e-03
     7:  2.1367e+00  2.1370e+00  2e-01  7e-04  7e-02  1e-03
     8:  2.1384e+00  2.1385e+00  1e-01  4e-04  5e-02  6e-04
     9:  2.1277e+00  2.1277e+00  2e-02  6e-05  7e-03  8e-05
    10:  2.1269e+00  2.1269e+00  2e-03  6e-06  7e-04  8e-06
    11:  2.1269e+00  2.1269e+00  3e-05  1e-07  1e-05  1e-07
    12:  2.1269e+00  2.1269e+00  3e-07  1e-09  1e-07  1e-09
    13:  2.1269e+00  2.1269e+00  3e-09  1e-11  1e-09  1e-11
    Optimal solution found.


Get upper and lower bound of model on the training data set:


```python
upper_bound, lower_bound = model.predict(x)
```


Plot the training data with the model predictions:


```python
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:, 0], x[:, 1], y, color='black')
ax.scatter(x[:, 0], x[:, 1], lower_bound[:, 0])
ax.scatter(x[:, 0], x[:, 1], upper_bound[:, 0])
```




    <mpl_toolkits.mplot3d.art3d.Path3DCollection at 0x121a35eb0>




![png](output_11_1.png)


The fake training data is plotted in black. The upper bound predicted by our model is shown in yellow, and the lower bound is shown in blue.



```python
model.get_model_reliability()
```




    0.6817243122768559



This means that, for a test set generated from the same function as the training data, our model predictions will be enclose the test set with probability greater than 0.682.
