"""
This module contains code to maximize the log likelihood function by changing parameter A.
"""

import numpy as np
import matplotlib.pyplot as plt


def H_generator(M):
    """
    This function generates the H matrix for a given M value.
    :param M: Number of neurons
    :return: H matrix
    """
    K = int(M * (M - 1) / 2)
    H = np.zeros((K, M, M))
    a = 1
    for i in range(0, M - 1):
        while a <= (K - ((M - 1 - i) * (M - 2 - i) / 2)):
            a_0 = (K - ((M - i) * (M - i - 1) / 2)) + 1
            indexer = int(a - a_0 + 1 + i)
            H[a - 1][i][indexer] = 1
            H[a - 1][indexer][i] = -1
            a += 1
    return H


def A_matrix(Z):
    """
    This function calculates the A matrix
    :param Z: The Z matrix
    :param delta_z: The delta_z matrix
    :return: The A matrix
    """
    delta_Z = np.diff(Z, axis=2)
    dZshape = delta_Z.shape
    Zshape = Z.shape
    delta_z = np.reshape(delta_Z, (dZshape[0], dZshape[1] * dZshape[2]))
    print(delta_z.shape)
    Z = np.reshape(Z[:, :, 0:-1], (Zshape[0], Zshape[1] * Zshape[2] - Zshape[1]))

    H = H_generator(Z.shape[0])
    W = np.tensordot(H, Z, axes=1)
    b = np.tensordot(W, delta_z, axes=([1, 2], [0, 1]))
    Q = np.tensordot(W, W, axes=([1, 2], [1, 2]))

    Beta = np.linalg.solve(Q, b)

    A = np.tensordot(Beta, H, axes=1)
    return A


def main():
    # Load the data from the .pyz file
    data = np.load("Data/Exercise_2C.npz")
    Z = data["Z"]
    A = A_matrix(Z)

    # calculate the error between the A matrix and the A_test matrix
    # error = LA.norm(A - A_test)
    # print(f"Error: {error}")

    # Plot A as a color map
    fig, ax = plt.subplots()
    im = ax.imshow(A, cmap="RdBu_r")
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")
    ax.set_title("A matrix")
    fig.colorbar(im)
    plt.show()
    fig, ax = plt.subplots()


# # Plot the matrix elements using imshow and absolute values
#     cax = ax.imshow(np.abs(A), cmap='RdBu', interpolation='none', origin='upper')
#     size = A.shape[0]

#     # Add arrows to indicate the sign of the elements
#     for i in range(size):
#         for j in range(size):
#             if A[i, j] < 0:
#                 ax.arrow(j, i, 0.0, 0.1, head_width=0.1, head_length=0.1, fc='black', ec='black')
#             elif A[i, j] > 0:
#                 ax.arrow(j, i, 0.0, -0.1, head_width=0.1, head_length=0.1, fc='black', ec='black')

#     # Add a colorbar
#     cbar = fig.colorbar(cax, ax=ax, label='Absolute Value')

#     # Set axis labels and title
#     ax.set_xticks(np.arange(size))
#     ax.set_yticks(np.arange(size))
#     ax.set_xticklabels(np.arange(1, size + 1))
#     ax.set_yticklabels(np.arange(1, size + 1))
#     ax.set_xlabel('Column')
#     ax.set_ylabel('Row')
#     ax.set_title('Antisymmetric Matrix Visualization')

#     plt.show()

if __name__ == "__main__":
    main()
