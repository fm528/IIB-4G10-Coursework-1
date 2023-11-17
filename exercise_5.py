import exercise_4b
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import Data.cond_color as cc


def find_Z_proj(Z, i, time):
    A = exercise_4b.A_matrix(Z)

    # Calculate the eigenvalues and eigenvectors of A
    eigenvalues, eigenvectors = LA.eig(A)
    eigenvectors = eigenvectors[:, ::2]
    eigenvalues = eigenvalues[::2]
    idx = eigenvalues.imag.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Find P which is the eigenvector corresponding to the largest eigenvalue, remove conjugate normalise
    P = (
        np.stack(
            (
                (eigenvectors[:, i].real) / LA.norm(eigenvectors[:, i].real),
                eigenvectors[:, i].imag / LA.norm(eigenvectors[:, i].imag),
            ),
            axis=1,
        )
    ).T
    

    # project z onto p and plot
    # print(f"Z: {Z.shape}", f"P: {P.shape}")

    mask = (time >= -150) & (time <= 200)
    Z = Z[:, :, mask]
    ZShape = Z.shape
    Z = Z.reshape(ZShape[0], ZShape[1] * ZShape[2])

    Z_proj = P @ Z

    Z_proj = Z_proj.reshape(2, ZShape[1], ZShape[2])

    return Z_proj


def plot_trajectory(Z_proj):
    colors = cc.get_colors(Z_proj[0, :, 0], Z_proj[1, :, 0])

    # Plot the trajectories for all conditions in the same plot
    fig, ax = plt.subplots()
    for i in range(108):
        ax.plot(Z_proj[0, i, :], Z_proj[1, i, :], color=colors[i])
        cc.plot_start(Z_proj[0, i, 0], Z_proj[1, i, 0], colors[i], ax=ax, markersize=15)
        cc.plot_end(Z_proj[0, i, -1], Z_proj[1, i, -1], colors[i], ax=ax, markersize=10)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Trajectories in the PC1-PC2 plane")
    plt.show()


def main():
    data = np.load("Data/Exercise_2C.npz")
    Z = data["Z"]
    V_m = data["V"]
    time = data["times"]

    # plot the trajectories on the first FR plane
    Z_1 = find_Z_proj(Z, 0, time)
    plot_trajectory(Z_1)

    # plot the trajectories on the second FR plane
    Z_2 = find_Z_proj(Z, 1, time)
    plot_trajectory(Z_2)

    # plot the trajectories on the third FR plane
    Z_3 = find_Z_proj(Z, 2, time)
    plot_trajectory(Z_3)


if __name__ == "__main__":
    main()
