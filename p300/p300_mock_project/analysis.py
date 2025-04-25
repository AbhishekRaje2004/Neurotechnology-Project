
import numpy as np
import matplotlib.pyplot as plt

def ensemble_average(epochs, target_indices):
    return np.mean(epochs[target_indices], axis=0)

def plot_erp(avg_data, fs=250, title="ERP", trigger_time=0):
    timepoints = avg_data.shape[1]
    t = np.linspace(-0.2, 0.8, timepoints)

    plt.figure(figsize=(10, 5))
    plt.plot(t, avg_data[0], label='ERP (target)')
    plt.axvline(trigger_time, color='red', linestyle='--', label='Stimulus Trigger')
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (µV)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
