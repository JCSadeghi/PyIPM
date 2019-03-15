import numpy as np
from PyIPM import PyIPM

np.random.seed(0)

model = PyIPM(polynomial_degree=2)

x = 5 * (np.random.rand(100, 2) - 0.5)
y = x[:, 0] ** 2 + x[:, 1] ** 2 * np.random.rand(1, 100)

model.fit(x, y[0, :])

upper_bound, lower_bound = model.predict(x)

tolerance = 0.01

assert(upper_bound[0] - 1.15 < tolerance), "Upper bound does not match stored value"

assert(lower_bound[0] - 0.15 < tolerance), "Lower bound does not match stored value"

ModelReliability = model.get_model_reliability()

assert(ModelReliability - 0.88 < tolerance), "Model reliability does not match stored value"
