import exercise_2AB
import exercise_2C
import exercise_4b
import exercise_5
import exercise_6
import numpy as np
import matplotlib.pyplot as plt
import random

# Load data
data = np.load("Data/psths.npz")
X = data["X"]
times = data["times"]

# find the index of the -150 time point
idx = np.where(times == -150)[0][0]
print(idx)
print(f"time: {times[idx]}")
# choose half of all conditions randomly for each neuron and invert the firing rates according to the equation x[t0:T] = 2 * x[t0] - x[t0:T]
for i, values in enumerate(X):
    C = np.random.choice(108, (108//2,), replace=False)
    for j in C:
        X[i][j][idx:] = 2 * X[i][j][idx] - X[i][j][idx:]

x_normal = exercise_2AB.max_firing_rate(X)
X = exercise_2AB.remove_mean_firing_rate(x_normal)

time = data["times"]
mask = (time >= -150) & (time <= 300)
time = time[mask]
X = X[:, :, mask]
X_shape = X.shape
X = X.reshape(X_shape[0], X_shape[1] * X_shape[2])
Z, V_m = exercise_2C.find_V_m(X)

Z_proj, P = exercise_5.find_Z_proj(Z, 0, time)

exercise_5.plot_trajectory(Z_proj)
