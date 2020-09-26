# -*- coding: utf-8 -*-

import setuptools


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setuptools.setup(
    name='PyIPM',
    version='1.1',
    description='Interval Predictor Models in Python',
    long_description=readme,
    author='Jonathan Sadeghi',
    author_email='J.C.Sadeghi@liverpool.ac.uk',
    url='https://github.com/JCSadeghi/PyIPM/',
    license=license,
    py_modules=["PyIPM"],
    install_requires=[
        'numpy>=1.12.1',
        'cvxopt>=1.1.9',
        'scikit-learn>=0.18.1',
        'scipy>=0.19.0',
    ],
    tests_require=[
        'pytest',
    ],
    python_requires='>=3.7',
)