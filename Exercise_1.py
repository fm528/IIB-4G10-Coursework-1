import numpy as np
import matplotlib.pyplot as plt

with open("Data/psths.npz", "rb") as f:
    data = np.load(f)
    X, times = data["X"], data["times"]

# Plot PSTHs
fig, ax = plt.subplots()

# generate 5 random colours, and 5 different line styles
colors = (
    np.array(
        [
            [149, 247, 95],
            [52, 76, 224],
            [145, 175, 203],
            [114, 42, 64],
            [45, 247, 176],
        ]
    )
    / 255
)
line_styles = ["-", "--", "-.", ":", "-"]


# Plot 10 random PSTHs
z = np.random.choice(108, 5, replace=False)
q = np.random.choice(182, 5, replace=False)
for i, neuron in enumerate(q):
    for j, condition in enumerate(z):
        ax.plot(times, X[neuron][condition], color=colors[i], linestyle=line_styles[j])

# generate a key for the plot that maps colours to neurons and line styles to conditions
handles = []
labels = []
for i, neuron in enumerate(q):
    handles.append(plt.Line2D([0], [0], color=colors[i], lw=2))
    labels.append(f"Neuron {neuron}")
for j, condition in enumerate(z):
    handles.append(plt.Line2D([0], [0], color="black", lw=2, linestyle=line_styles[j]))
    labels.append(f"Condition {condition}")

# plot the key
ax.legend(handles, labels, ncol=5, fontsize="x-small", markerscale=0.5)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Firing rate (Hz)")
ax.set_title("PSTHs")
plt.show()

# Plot the population average firing rate
population_avg = np.mean(X, axis=(0, 1))

fig, ax = plt.subplots()
ax.plot(times, population_avg)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("Population average firing rate (Hz)")
ax.set_title("Population average firing rate over all neurons and conditions")
plt.show()
