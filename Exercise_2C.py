import numpy as np
from numpy import linalg as LA


def main():
    # Load the data from the .pyz file
    data = np.load("Data/psths_norm.npz")

    # Slice the data between time -150 and 300
    time = data["times"]
    mask = (time >= -150) & (time <= 300)
    X = data["X"][:, :, mask]
    X_shape = X.shape
    X = X.reshape(X_shape[0], X_shape[1] * X_shape[2])
    Z, V_m = find_V_m(X)
    # Save Z and V to a pyz file
    np.savez("Data/Exercise_2C.npz", Z=Z, V=V_m, times=time[mask])


def find_V_m(X):
    S = (1 / X.shape[1]) * X @ X.T
    print(f"S shape: {S.shape}")
    eigenvalues, eigenvectors = LA.eig(S)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Sort the eigenvalues and eigenvectors in descending order
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    eigenvectors = eigenvectors[:, idx]

    # normalize the eigenvectors
    eigenvectors = eigenvectors / LA.norm(eigenvectors, axis=0)

    # find V_m
    V_m = eigenvectors[:, :12]
    print(f"V_m: {V_m.shape}")

    # print the columns of V_m
    # print(f"V_m: {V_m[:, 0]}")

    # find the projection of the neurons onto the first 12 eigenvectors
    Z = V_m.T @ X
    Z = Z.reshape(12, 108, 46)

    return Z, V_m


if __name__ == "__main__":
    main()
