import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
# Load the firing rates data
data = np.load("Data/psths.npz")
firing_rates = data["X"]
times = data["times"]


# calculate the max firing rate for each neuron and plot histogram PART A
def max_firing_rate(fr):
    normalised_firing_rates = np.zeros_like(fr)
    a = fr.max(axis=(1, 2))
    b = fr.min(axis=(1, 2))
    for i, values in enumerate(a):
        normalised_firing_rates[i] = [(z - b[i]) / (values - b[i] + 5) for z in fr[i]]
    return normalised_firing_rates


# Remove from X the cross-condition mean firing rate for each neuron and time PART B
def remove_mean_firing_rate(normalised_firing_rates):
    mu = np.mean(normalised_firing_rates, axis=1)
    # print(f"mean: {mean.shape}")
    # print(f"normalised Firing Rates: {normalised_firing_rates.shape}")
    new_normal = np.moveaxis(normalised_firing_rates, 1, 0)
    # print(f"normalised Firing Rates: {new_normal.shape}")
    for i, values in enumerate(new_normal):
        new_normal[i] = values - mu
    normalised_firing_rates = np.moveaxis(new_normal, 0, 1)
    return normalised_firing_rates

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

def main():
    # pop_average_firing_rate(firing_rates)
    normalised_firing_rates = max_firing_rate(firing_rates)
    a = firing_rates.max(axis=(1, 2))
    _, ax = plt.subplots()
    ax.hist(a, bins=20)
    ax.set_xlabel("Maximum firing rate (Hz)")
    ax.set_ylabel("Number of neurons")
    ax.set_title("Distribution of maximum firing rates")
    plt.show()

    normalised_firing_rates = remove_mean_firing_rate(normalised_firing_rates)
    # Plot the normaLised firing rates for all neurons under condition 1 DEBUG
    _, ax = plt.subplots()
    colours = (
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

    z = [86, 80, 6, 18, 107]
    q = [177, 144, 99, 71, 80]
    for i, neuron in enumerate(q):
        for j, condition in enumerate(z):
            ax.plot(
                times,
                normalised_firing_rates[neuron][condition],
                color=colours[i],
                linestyle=line_styles[j],
            )

    # generate a key for the plot that maps colours to neurons and line styles to conditions
    handles = []
    labels = []
    for i, neuron in enumerate(q):
        handles.append(plt.Line2D([0], [0], color=colours[i], lw=2))
        labels.append(f"Neuron {neuron}")
    for j, condition in enumerate(z):
        handles.append(
            plt.Line2D([0], [0], color="black", lw=2, linestyle=line_styles[j])
        )
        labels.append(f"Condition {condition}")

    # plot the key
    ax.legend(handles, labels, ncol=5, fontsize="x-small", markerscale=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.set_title("Normalized PSTHs")
    plt.show()

    # save the normalised data in  'Data\psths_norm.npz'
    np.savez("Data/psths_norm.npz", X=normalised_firing_rates, times=times)

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


if __name__ == "__main__":
    main()

