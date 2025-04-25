
import numpy as np

def extract_epochs(eeg, timestamps, events, fs=250, pre=0.2, post=0.8):
    eeg = np.array(eeg)
    ts = np.array(timestamps)
    pre_samp = int(pre * fs)
    post_samp = int(post * fs)

    epochs = []
    labels = []

    for evt_time, label in events:
        idx = np.searchsorted(ts, evt_time)
        if idx - pre_samp >= 0 and idx + post_samp < len(eeg):
            epoch = eeg[idx - pre_samp : idx + post_samp].T
            epochs.append(epoch)
            labels.append(label)

    return np.array(epochs), labels
