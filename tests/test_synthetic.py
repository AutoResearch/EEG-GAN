
# Load modules
import numpy as np
import matplotlib.pyplot as plt

# Create simulated data
x = np.linspace(-1, 3.95, 100)  # Create variable


# Create model
def quinticModel(x, scale=5):
    x = np.array(x, dtype=complex)

    # This will be used to supress the beginning and end of the time series
    hanningStart = np.hanning(len(x) / 2)
    hanningEnd = np.hanning(len(x))
    sup = np.ones(len(x))
    sup[0:int(len(hanningStart) / 2)] = hanningStart[0:int(len(hanningStart) / 2)]
    sup[-int(len(hanningEnd) / 2):] = hanningEnd[-int(len(hanningEnd) / 2):]

    # This will be used to zero the baseline
    baseline = x.copy()
    baselineHanning = np.hanning(len(baseline[baseline > 0]))
    baselineHanning[int(len(baselineHanning) / 2):] = 1
    baseline[baseline < 0] = 0
    baseline[baseline > 0] = baselineHanning

    # The raw model
    y = pow(x, 5) - scale * pow(x, 4) + scale * pow(x, 3) + scale * pow(x, 2) - 6 * x - 1

return -(baseline * (np.flip(sup * -y))).real


# Create a load of samples per condition
noise = 5
size = 1000
condition00 = np.asarray(
    [np.insert(quinticModel(x, 5 + np.random.rand(1) * noise), 0, np.concatenate(([0], [0]))) for row in range(size)])
condition01 = np.asarray(
    [np.insert(quinticModel(x, 10 + np.random.rand(1) * noise), 0, np.concatenate(([0], [1]))) for row in range(size)])
condition10 = np.asarray(
    [np.insert(quinticModel(x, 0 + np.random.rand(1) * noise), 0, np.concatenate(([1], [0]))) for row in range(size)])
condition11 = np.asarray(
    [np.insert(quinticModel(x, -5 + np.random.rand(1) * noise), 0, np.concatenate(([1], [1]))) for row in range(size)])

# Combine all conditions
data = np.vstack((condition00, condition01, condition10, condition11))
data = np.insert(data, 0, np.tile(np.arange(1, size + 1), 4), 1)  # Add 'participantIDs'
data = np.insert(data, 2, np.repeat(np.arange(1, 5), size), 1)  # Add 'trial'
np.random.shuffle(data)

# Save as csv
columnNames = ['ParticipantID', 'Condition', 'Trial', 'Electrode']

for n in range(data.shape[1] - 4):
    columnNames.append('Time' + str(n + 1))
np.savetxt('gansMultiCondition.csv', data, delimiter=',', header=','.join(columnNames), comments='')

# Plot individual samples
plot_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
for row in range(len(condition00)):
    plt.plot(x, condition00[row, 2:], alpha=.008, color=plot_colors[0])
    plt.plot(x, condition01[row, 2:], alpha=.008, color=plot_colors[1])
    plt.plot(x, condition10[row, 2:], alpha=.008, color=plot_colors[2])
    plt.plot(x, condition11[row, 2:], alpha=.008, color=plot_colors[3])

# Plot averaged data per condition
plt.plot(x, np.mean(condition00[:, 2:], axis=0), color=plot_colors[0], label='Condition00')
plt.plot(x, np.mean(condition01[:, 2:], axis=0), color=plot_colors[1], label='Condition01')
plt.plot(x, np.mean(condition10[:, 2:], axis=0), color=plot_colors[2], label='Condition10')
plt.plot(x, np.mean(condition11[:, 2:], axis=0), color=plot_colors[3], label='Condition11')

# Format and show plot
plt.legend(frameon=False)
plt.title('Ground Truth')
plt.show()