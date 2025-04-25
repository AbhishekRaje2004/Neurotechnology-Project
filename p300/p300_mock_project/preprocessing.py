import numpy as np
from scipy import signal

def preprocess_eeg(eeg_data, fs=250, notch_freq=50, bandpass=(1, 30), normalize=True):
    """
    Preprocess EEG data with filters and normalization
    
    Parameters:
    -----------
    eeg_data : array-like
        Raw EEG data, shape (n_channels, n_samples) or (n_samples,) for single channel
    fs : float
        Sampling frequency in Hz
    notch_freq : float or None
        Frequency to apply notch filter (typically line noise at 50 or 60 Hz)
    bandpass : tuple or None
        (low, high) cutoff frequencies for bandpass filter
    normalize : bool
        Whether to z-score normalize the data
        
    Returns:
    --------
    processed_data : array-like
        Processed EEG data with same shape as input
    """
    # Ensure data is numpy array
    data = np.asarray(eeg_data)
    
    # Handle dimensions
    if data.ndim == 1:
        data = data.reshape(1, -1)  # Convert to (1, n_samples)
    
    n_channels, n_samples = data.shape
    processed = np.zeros_like(data)
    
    for ch in range(n_channels):
        # Apply notch filter for line noise (if requested)
        if notch_freq is not None:
            # Enhanced notch filter with wider bandwidth for more effective line noise removal
            q_factor = 30  # Quality factor
            b_notch, a_notch = signal.iirnotch(notch_freq, q_factor, fs)
            data_notched = signal.filtfilt(b_notch, a_notch, data[ch])
            
            # Apply additional notch at harmonics (100Hz) if within Nyquist frequency
            if 2 * notch_freq < fs / 2:
                b_notch2, a_notch2 = signal.iirnotch(2 * notch_freq, q_factor, fs)
                data_notched = signal.filtfilt(b_notch2, a_notch2, data_notched)
        else:
            data_notched = data[ch]
            
        # Apply bandpass filter (if requested)
        if bandpass is not None:
            low, high = bandpass
            # Higher order filter (6) for steeper roll-off
            b_band, a_band = signal.butter(6, [low, high], btype='band', fs=fs)
            data_filtered = signal.filtfilt(b_band, a_band, data_notched)
        else:
            data_filtered = data_notched
            
        # Store filtered data
        processed[ch] = data_filtered
    
    # Apply z-score normalization
    if normalize:
        for ch in range(n_channels):
            mean = np.mean(processed[ch])
            std = np.std(processed[ch])
            if std > 0:  # Avoid division by zero
                processed[ch] = (processed[ch] - mean) / std
            
    # Return in the same shape as input
    if eeg_data.ndim == 1:
        return processed.squeeze()
    return processed

def apply_adaptive_filter(eeg_data, fs=250, filter_type='savgol', window_len=101, poly_order=3):
    """
    Apply adaptive filtering to remove noise while preserving signal features
    
    Parameters:
    -----------
    eeg_data : array-like
        EEG data, shape (n_channels, n_samples) or (n_samples,)
    fs : float
        Sampling frequency in Hz
    filter_type : str
        'savgol' for Savitzky-Golay filter or 'median' for median filter
    window_len : int
        Window length for the filter
    poly_order : int
        Polynomial order for Savitzky-Golay filter
        
    Returns:
    --------
    filtered_data : array-like
        Filtered data with same shape as input
    """
    # Ensure data is numpy array
    data = np.asarray(eeg_data)
    orig_shape = data.shape
    
    # Handle dimensions
    if data.ndim == 1:
        data = data.reshape(1, -1)  # Convert to (1, n_samples)
    
    n_channels, n_samples = data.shape
    filtered = np.zeros_like(data)
    
    for ch in range(n_channels):
        if filter_type == 'savgol':
            # Ensure window length is odd
            if window_len % 2 == 0:
                window_len += 1
            filtered[ch] = signal.savgol_filter(data[ch], window_len, poly_order)
        elif filter_type == 'median':
            filtered[ch] = signal.medfilt(data[ch], window_len)
        else:
            filtered[ch] = data[ch]  # No filtering if invalid type
    
    # Return in the same shape as input
    if len(orig_shape) == 1:
        return filtered.squeeze()
    return filtered

def remove_line_noise(eeg_data, fs=250, line_freq=50, harmonics=3):
    """
    Remove power line noise (50Hz or 60Hz) and its harmonics
    
    Parameters:
    -----------
    eeg_data : array-like
        EEG data, shape (n_channels, n_samples) or (n_samples,)
    fs : float
        Sampling frequency in Hz
    line_freq : float
        Line frequency to remove (typically 50Hz in Europe/Asia, 60Hz in Americas)
    harmonics : int
        Number of harmonics to remove
        
    Returns:
    --------
    cleaned_data : array-like
        Cleaned data with same shape as input
    """
    # Ensure data is numpy array
    data = np.asarray(eeg_data)
    orig_shape = data.shape
    
    # Handle dimensions
    if data.ndim == 1:
        data = data.reshape(1, -1)  # Convert to (1, n_samples)
    
    n_channels, n_samples = data.shape
    cleaned = data.copy()
    
    # Quality factor for the notch filters
    q = 30
    
    for ch in range(n_channels):
        # Apply notch filters at the base frequency and its harmonics
        for harm in range(1, harmonics+1):
            freq = harm * line_freq
            if freq < fs/2:  # Only apply if below Nyquist frequency
                b, a = signal.iirnotch(freq, q, fs)
                cleaned[ch] = signal.filtfilt(b, a, cleaned[ch])
    
    # Return in the same shape as input
    if len(orig_shape) == 1:
        return cleaned.squeeze()
    return cleaned

def segment_data(eeg_data, markers, fs=250, pre_stim=0.2, post_stim=0.8):
    """
    Extract data segments around markers
    
    Parameters:
    -----------
    eeg_data : array-like
        EEG data, shape (n_channels, n_samples)
    markers : list of tuples
        List of (timestamp, marker_code) tuples
    fs : float
        Sampling frequency in Hz
    pre_stim : float
        Time before stimulus in seconds
    post_stim : float
        Time after stimulus in seconds
        
    Returns:
    --------
    segments : array-like
        Segmented data with shape (n_segments, n_channels, n_samples_per_segment)
    """
    # Calculate number of samples for each segment
    pre_samples = int(pre_stim * fs)
    post_samples = int(post_stim * fs)
    total_samples = pre_samples + post_samples
    
    # Initialize segments array
    n_segments = len(markers)
    n_channels = eeg_data.shape[0] if eeg_data.ndim > 1 else 1
    
    segments = np.zeros((n_segments, n_channels, total_samples))
    
    # Extract segments
    for i, (marker_time, _) in enumerate(markers):
        # Find sample index closest to the marker timestamp
        marker_idx = int(marker_time * fs)
        
        # Calculate segment indices
        start_idx = marker_idx - pre_samples
        end_idx = marker_idx + post_samples
        
        # Skip segments that would go out of bounds
        if start_idx < 0 or end_idx >= eeg_data.shape[-1]:
            segments[i] = np.nan
            continue
            
        # Extract segment
        if eeg_data.ndim > 1:
            segments[i] = eeg_data[:, start_idx:end_idx]
        else:
            segments[i, 0] = eeg_data[start_idx:end_idx]
    
    # Remove NaN segments
    valid_mask = ~np.isnan(segments).any(axis=(1, 2))
    return segments[valid_mask]