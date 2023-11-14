
import numpy as np
import matplotlib.pyplot as plt
from Data.cond_color import get_colors, plot_start, plot_end

# Load data
data = np.load('Data/Exercise_2C.npz')
Z = data['Z']
time = data['times']

# Get colors for each condition
colors = get_colors(Z[0, :, 0], Z[1, :, 0])

# Plot the trajectories for all conditions in the same plot
fig, ax = plt.subplots()
for i in range(108):
    ax.plot(Z[0, i, :], Z[1, i, :], color=colors[i])
    plot_start(Z[0, i, 0], Z[1, i, 0], colors[i], ax=ax, markersize=15)
    plot_end(Z[0, i, -1], Z[1, i, -1], colors[i], ax=ax, markersize=10)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_title("Trajectories in the PC1-PC2 plane")
plt.show()