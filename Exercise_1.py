import numpy as np
import matplotlib.pyplot as plt

with open("Data/psths.npz", "rb") as f:
    data = np.load(f)
    X, times = data["X"], data["times"]

# Plot PSTHs
fig, ax = plt.subplots()

# Plot PSTHs for all neurons under condition 1
for i in range(182):
    ax.plot(times, X[i][0], label=f"Neuron {i+1}: Condition 1")

ax.set_xlabel("Time (ms)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("PSTHs")
plt.show()
