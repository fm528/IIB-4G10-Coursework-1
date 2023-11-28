import matplotlib.pyplot as plt
import numpy as np
import Data.cond_color as cc


def main():
    # Load data
    data = np.load("Data/Exercise_2C.npz")
    Z = data["Z"]

    # Plot the trajectories
    plot_trajectory(Z)


def plot_trajectory(Z, alt=False):
    # Get colors for each condition
    colors = cc.get_colors(Z[0, :, 0], Z[1, :, 0], alt_colors=alt)

    # Plot the trajectories for all conditions in the same plot
    _, ax = plt.subplots()
    for i in range(108):
        ax.plot(Z[0, i, :], Z[1, i, :], color=colors[i])
        cc.plot_start(Z[0, i, 0], Z[1, i, 0], colors[i], ax=ax, markersize=15)
        cc.plot_end(Z[0, i, -1], Z[1, i, -1], colors[i], ax=ax, markersize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Trajectories in the PC1-PC2 plane")
    plt.show()


if __name__ == "__main__":
    main()
