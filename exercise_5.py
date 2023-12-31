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

    return Z_proj, P


def plot_trajectory(Z_proj, title, alt=False):
    colors = cc.get_colors(Z_proj[0, :, 0], Z_proj[1, :, 0], alt_colors=alt)

    # Plot the trajectories for all conditions in the same plot
    _, ax = plt.subplots()
    for i in range(108):
        ax.plot(Z_proj[0, i, :], Z_proj[1, i, :], color=colors[i])
        if (
            i == 0
        ):  # add labels only for the first plot to avoid duplicate legend entries
            cc.plot_start(
                Z_proj[0, i, 0],
                Z_proj[1, i, 0],
                colors[i],
                ax=ax,
                markersize=20,
                label=1,
            )
            cc.plot_end(
                Z_proj[0, i, -1],
                Z_proj[1, i, -1],
                colors[i],
                ax=ax,
                markersize=10,
                label=1,
            )
        else:
            cc.plot_start(
                Z_proj[0, i, 0], Z_proj[1, i, 0], colors[i], ax=ax, markersize=20
            )
            cc.plot_end(
                Z_proj[0, i, -1], Z_proj[1, i, -1], colors[i], ax=ax, markersize=10
            )
    ax.set_xlabel("real")
    ax.set_ylabel("imag")
    ax.set_title(title)
    plt.legend()  # add a legend
    plt.show()


def main():
    data = np.load("Data/Exercise_2C.npz")
    Z = data["Z"]
    time = data["times"]

    # plot the trajectories on the first FR plane
    Z_1, P_fr = find_Z_proj(Z, 0, time)
    print(f"Z_1: {Z_1.shape}")
    # plot_trajectory(Z_1, "Plane of 1st FR")

    # plot the trajectories on the second FR plane
    Z_2, P_fr1 = find_Z_proj(Z, 1, time)
    # plot_trajectory(Z_2, "Plane of 2nd FR")

    # plot the trajectories on the third FR plane
    Z_3, P_fr2 = find_Z_proj(Z, 2, time)
    # plot_trajectory(Z_3, "Plane of 3rd FR")

    np.savez("Data/exercise_5.npz", P=[P_fr, P_fr1, P_fr2], Z_1=Z_1, Z_2=Z_2, Z_3=Z_3)


if __name__ == "__main__":
    main()
