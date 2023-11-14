"""
This module contains code to maximize the log likelihood function by changing parameter A.
"""

import numpy as np

data = np.load('Data/Exercise_2C.npz')
Z = data['Z']
time = data['times']

# maximise log likelihood function by changing parameter A
A = np.random.rand(12, 12)
log_likelihood = np.sum(Z * np.log(A * time) - A * time)
print(f"log likelihood: {log_likelihood}")
