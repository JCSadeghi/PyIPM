# -*- coding: utf-8 -*-
import setuptools

setuptools.setup(
    name="PyIPM",
    version="2.0",
    description="Interval Predictor Models in Python",
    author="Jonathan Sadeghi",
    author_email="J.C.Sadeghi@liverpool.ac.uk",
    url="https://github.com/JCSadeghi/PyIPM/",
    license="LGPL-3.0 License",
    py_modules=["PyIPM"],
    install_requires=[
        "numpy>=1.12.1",
        "cvxopt>=1.1.9",
        "scikit-learn>=0.18.1",
        "scipy>=0.19.0",
    ],
    tests_require=["pytest"],
    python_requires=">=3.7",
)
