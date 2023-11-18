import exercise_4b
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import Data.cond_color as cc
import exercise_5

def main():
    data_1 = np.load("Data/Exercise_2C.npz")
    Z = data_1["Z"]
    V_m = data_1["V"]

    data_2 = np.load("Data/exercise_5.npz")
    P = data_2["P"]
    Z = data_2["Z_1"]
    print(f"Z: {Z.shape}")

    data_3 = np.load("Data/psths_norm.npz")

    X = data_3["X"]
    times = data_3["times"]


# select only the data between -800, -150
    mask = (times >= -800) & (times <= -150)
    times = times[mask]
    X = X[:, :, mask]
    X_shape = X.shape

    Z_n = V_m.T @ X.reshape(X_shape[0], X_shape[1] * X_shape[2])
    Z_proj = P @ Z_n
    Z_proj = Z_proj.reshape(2, X_shape[1], X_shape[2])

    colors = cc.get_colors(Z_proj[0, :, -1], Z_proj[1, :, -1], alt_colors=1)

# Plot the trajectories for all conditions in the same plot
    fig, ax = plt.subplots()
    for i in range(108):
        ax.plot(Z_proj[0, i, :], Z_proj[1, i, :], color=colors[i])
        cc.plot_start(Z_proj[0, i, 0], Z_proj[1, i, 0], colors[i], ax=ax, markersize=15)
        cc.plot_end(Z_proj[0, i, -1], Z_proj[1, i, -1], colors[i], ax=ax, markersize=10)
    

    colors = cc.get_colors(Z[0, :, 0], Z[1, :, 0])

# Plot the trajectories for all conditions in the same plot
    for i in range(108):
        ax.plot(Z[0, i, :], Z[1, i, :], color=colors[i], alpha = 0.25)
        # cc.plot_start(Z[0, i, 0], Z[1, i, 0], colors[i], ax=ax, markersize=15)
        cc.plot_end(Z[0, i, -1], Z[1, i, -1], colors[i], ax=ax, markersize=10)
    ax.set_xlabel("real")
    ax.set_ylabel("imag")
    ax.set_title("Plane of fastest rotation")
    plt.show()

if __name__ == "__main__":
    main()
