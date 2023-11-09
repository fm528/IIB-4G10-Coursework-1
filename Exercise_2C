import numpy as np
from numpy import linalg as LA
import scipy.io as sio
import matplotlib.pyplot as plt
import Data.cond_color

# Load the data from the .pyz file
data = np.load('Data/psths_norm.npz')

# Slice the data between time -150 and 300
time = data['times']
mask = (time >= -150) & (time <= 300)
X = data['X'][:,:, mask]
X_shape = X.shape
print(f"X shape: {X.shape}")
X = X.reshape(X_shape[0], X_shape[1] * X_shape[2])
print(f"X shape: {X.shape}")

S = (1/X.shape[1]) * X @ X.T
print(f"S shape: {S.shape}")
eigenvalues, eigenvectors = LA.eig(S)
eigenvalues = eigenvalues.real
eigenvectors = eigenvectors.real

# Sort the eigenvalues and eigenvectors in descending order
idx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
eigenvectors = eigenvectors[:, idx]

# find V_m 
V_m = eigenvectors[:12].T
print(f"V_m shape: {V_m.shape}")

# print the columns of V_m
# print(f"V_m: {V_m[:, 0]}")

#find the projection of the neurons onto the first 12 eigenvectors
Z = V_m.T @ X
print(f"X_proj shape: {Z.shape}")
Z = Z.reshape(12, 108, 46)
print(f"X_proj shape: {Z.shape}")

# Generate the colors for the neurons
colors = Data.cond_color.get_colors(Z[0,:, 0], Z[1, :, 0], alt_colors=True)


# Plot the projection of the neurons onto the first 2 eigenvectors. Connect the points
# corresponding to the same condition with a line. Use the colors generated above.
fig, ax = plt.subplots()
for i in range(108):
    ax.plot(Z[0, i, :], Z[1, i, :], color=colors[i])
ax.set_xlabel("First eigenvector")
ax.set_ylabel("Second eigenvector")
ax.set_title("Projection of the neurons onto the first 2 eigenvectors")

# add start and end points
Data.cond_color.plot_start(Z[0, :, 0], Z[1, :, 0], colors, ax=ax, markersize=100)
Data.cond_color.plot_end(Z[0, :, -1], Z[1, :, -1], colors, ax=ax, markersize=20)
plt.show()

# Save Z and V to a pyz file
np.savez('Data/Exercise_2C.npz', Z=Z, V=V_m)
