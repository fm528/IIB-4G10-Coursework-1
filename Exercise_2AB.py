import matplotlib.pyplot as plt
import numpy as np

# Load the firing rates data
data = np.load('Data/psths.npz')
firing_rates = data['X']
times = data['times']


#calculate the max firing rate for each neuron and plot histogram PART A

normalised_firing_rates = np.zeros_like(firing_rates)
a = firing_rates.max(axis=(1, 2))
b = firing_rates.min(axis=(1, 2))
for i, values in enumerate(a):
    # print(f"firing rates {i}: {firing_rates[i]}")
    # print(f"max {i}: {values}")
    # print(f"min {i}: {b[i]}")
    normalised_firing_rates[i] = [(z - b[i]) / (values - b[i] + 5) for z in firing_rates[i]]

fig, ax = plt.subplots()
ax.hist(a, bins=20)
ax.set_xlabel('Maximum firing rate (Hz)')
ax.set_ylabel('Number of neurons')
ax.set_title('Distribution of maximum firing rates')
plt.show()

# Plot the normaLised firing rates for all neurons under condition 1 DEBUG
# fig, ax = plt.subplots()
# for i in range(182):
#     ax.plot(times, normalised_firing_rates[i][0], label=f'Neuron {i+1}: Condition 1')
# ax.set_xlabel('Time (ms)')
# ax.set_ylabel('Firing rate (Hz)')
# ax.set_title('Normalized PSTHs')
# plt.show()

# print(normalised_firing_rates[67])
# Remove from X the cross-condition mean firing rate for each neuron and time PART B

mu = normalised_firing_rates.mean(axis=(1))
# print(f"mean: {mean.shape}")
# print(f"normalised Firing Rates: {normalised_firing_rates.shape}")
new_normal = np.moveaxis(normalised_firing_rates, 1, 0)
# print(f"normalised Firing Rates: {new_normal.shape}")
for i, values in enumerate(new_normal):
    new_normal[i] = values - mu
# normalised_firing_rates = np.moveaxis(new_normal, 0, 1)
# print(f"normalised Firing Rates: {normalised_firing_rates.shape}")
# print(normalised_firing_rates[67])


# Plot the normaLised firing rates for all neurons under condition 1 DEBUG
fig, ax = plt.subplots()
for i in range(108):
    ax.plot(times, normalised_firing_rates[0][i], label=f'Neuron {i+1}: Condition 1')
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Firing rate (Hz)')
ax.set_title('Normalized PSTHs')
plt.show()

#save the normalised data in  'Data\psths_norm.npz'
np.savez('Data/psths_norm.npz', X=normalised_firing_rates, times=times)
