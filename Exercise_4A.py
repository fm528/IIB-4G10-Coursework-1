import numpy as np

data = np.load('Data/Exercise_2C.npz')
Z = data['Z']
time = data['times']

# maximise log likelihood function by changing parameter A
A = np.random.rand(12, 12)
print(f"A: {A}")

f