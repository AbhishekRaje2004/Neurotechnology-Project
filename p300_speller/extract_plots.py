#!/usr/bin/env python
# Extract plots and figures from PKL files for LaTeX report

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures

# Set paths
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "latex_report", "figures")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def load_pkl_file(filename):
    """Load a PKL file and return its contents"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_all_pkl_files():
    """Get list of all PKL files in the data directory"""
    pkl_files = []
    for file in os.listdir(data_dir):
        if file.endswith('.pkl'):
            pkl_files.append(os.path.join(data_dir, file))
    return pkl_files

def extract_model_metrics(model_file):
    """Extract model metrics from model file and create visualization"""
    try:
        model_data = load_pkl_file(model_file)
        
        if 'metrics' in model_data:
            metrics = model_data['metrics']
            
            # Create bar chart of performance metrics
            plt.figure(figsize=(10, 6))
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            plt.bar(metric_names, metric_values, color='steelblue')
            plt.title('Model Performance Metrics')
            plt.ylabel('Score')
            plt.ylim(0, 1.0)
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.xticks(rotation=45, ha='right')
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'model_metrics.png'), dpi=300)
            plt.close()
            
            # Create confusion matrix visualization if available
            if 'confusion_matrix' in model_data:
                cm = model_data['confusion_matrix']
                
                plt.figure(figsize=(8, 6))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                
                classes = ['Non-Target', 'Target']
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes)
                plt.yticks(tick_marks, classes)
                
                # Add text annotations to the confusion matrix
                thresh = cm.max() / 2.0
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                horizontalalignment="center",
                                color="white" if cm[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=300)
                plt.close()
            
            return True
        return False
    except Exception as e:
        print(f"Error extracting model metrics: {e}")
        return False

def extract_erp_plots(session_file):
    """Extract ERP plots from session files"""
    try:
        session_data = load_pkl_file(session_file)
        file_basename = os.path.basename(session_file).replace('.pkl', '')
        
        # Get ERP data if available
        if 'epochs' in session_data and 'target_indices' in session_data:
            epochs = session_data['epochs']
            target_indices = session_data['target_indices']
            
            if len(epochs) == 0:
                return False
                
            # Create target and non-target sets
            target_epochs = [epochs[i] for i in target_indices if i < len(epochs)]
            non_target_indices = [i for i in range(len(epochs)) if i not in target_indices]
            non_target_epochs = [epochs[i] for i in non_target_indices if i < len(epochs)]
            
            if len(target_epochs) == 0 or len(non_target_epochs) == 0:
                return False
            
            # Average ERPs
            avg_target = np.mean(target_epochs, axis=0)
            avg_non_target = np.mean(non_target_epochs, axis=0)
            
            # Get time vector if available, otherwise create one
            fs = session_data.get('sampling_rate', 256)  # Default to 256 Hz if not specified
            if 'times' in session_data:
                times = session_data['times']
            else:
                # Create time vector based on epoch length
                epoch_len = avg_target.shape[1]
                times = np.linspace(-0.2, 0.8, epoch_len)  # Assuming -200ms to 800ms window
            
            # Plot ERPs for each channel
            n_channels = avg_target.shape[0]
            channel_names = session_data.get('channel_names', [f"Ch{i+1}" for i in range(n_channels)])
            
            # Plot grand average ERP (average of all channels)
            plt.figure(figsize=(10, 6))
            
            grand_avg_target = np.mean(avg_target, axis=0)
            grand_avg_non_target = np.mean(avg_non_target, axis=0)
            
            plt.plot(times, grand_avg_target, 'b-', linewidth=2, label='Target')
            plt.plot(times, grand_avg_non_target, 'r-', linewidth=2, label='Non-Target')
            
            # Add shaded area for P300 window (typically 250-500ms)
            p300_start = 0.25
            p300_end = 0.5
            p300_start_idx = np.argmin(np.abs(times - p300_start))
            p300_end_idx = np.argmin(np.abs(times - p300_end))
            
            plt.axvspan(p300_start, p300_end, color='yellow', alpha=0.3)
            
            # Add vertical line at t=0 (stimulus onset)
            plt.axvline(x=0, color='k', linestyle='--')
            
            # Calculate P300 amplitude (max value in P300 window)
            p300_amp_target = np.max(grand_avg_target[p300_start_idx:p300_end_idx])
            p300_amp_non_target = np.max(grand_avg_non_target[p300_start_idx:p300_end_idx])
            
            # Add text annotation with P300 amplitude
            plt.annotate(f'P300 Amplitude = {p300_amp_target:.2f} μV', 
                        xy=(p300_end, p300_amp_target),
                        xytext=(p300_end + 0.1, p300_amp_target),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
            
            plt.grid(True)
            plt.title(f'Grand Average ERP ({file_basename})')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (μV)')
            plt.legend()
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'grand_avg_erp_{file_basename}.png'), dpi=300)
            plt.close()
            
            # Create a topographical plot if we have position information
            if 'channel_positions' in session_data and len(session_data['channel_positions']) == n_channels:
                try:
                    # Find peak time of P300
                    p300_peak_idx = p300_start_idx + np.argmax(grand_avg_target[p300_start_idx:p300_end_idx])
                    p300_peak_time = times[p300_peak_idx]
                    
                    # Get amplitudes at peak time for each channel
                    target_amplitudes = avg_target[:, p300_peak_idx]
                    non_target_amplitudes = avg_non_target[:, p300_peak_idx]
                    
                    # Difference wave
                    diff_amplitudes = target_amplitudes - non_target_amplitudes
                    
                    # Create topographical plot
                    plt.figure(figsize=(10, 8))
                    plt.subplot(111)
                    
                    # Extract positions
                    positions = np.array(session_data['channel_positions'])
                    x = positions[:, 0]
                    y = positions[:, 1]
                    
                    # Simple topographical plot using scatter with interpolation
                    plt.scatter(x, y, c=diff_amplitudes, cmap='jet', s=100)
                    plt.colorbar(label='Amplitude Difference (μV)')
                    
                    # Add channel labels
                    for i, label in enumerate(channel_names):
                        plt.annotate(label, (x[i], y[i]), fontsize=10, 
                                   ha='center', va='center')
                    
                    plt.title(f'Topographical Distribution at P300 Peak ({p300_peak_time:.3f}s)')
                    plt.axis('equal')
                    plt.grid(False)
                    plt.axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'topo_p300_{file_basename}.png'), dpi=300)
                    plt.close()
                except Exception as e:
                    print(f"Error creating topographical plot: {e}")
            
            # Create channel-wise ERP plot for key channels (Pz, Cz, P3, P4)
            key_channels = ['Pz', 'Cz', 'P3', 'P4']
            key_indices = []
            
            # Find indices of key channels if they exist
            for ch in key_channels:
                if ch in channel_names:
                    key_indices.append(channel_names.index(ch))
            
            if len(key_indices) > 0:
                plt.figure(figsize=(12, 8))
                for i, idx in enumerate(key_indices):
                    plt.subplot(2, 2, i+1)
                    plt.plot(times, avg_target[idx], 'b-', linewidth=2, label='Target')
                    plt.plot(times, avg_non_target[idx], 'r-', linewidth=2, label='Non-Target')
                    plt.axvspan(p300_start, p300_end, color='yellow', alpha=0.3)
                    plt.axvline(x=0, color='k', linestyle='--')
                    plt.grid(True)
                    plt.title(f'Channel: {channel_names[idx]}')
                    plt.xlabel('Time (s)')
                    plt.ylabel('Amplitude (μV)')
                    if i == 0:
                        plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'channel_erps_{file_basename}.png'), dpi=300)
                plt.close()
            
            # Time-frequency analysis for a key channel (e.g., Pz)
            if 'Pz' in channel_names:
                pz_idx = channel_names.index('Pz')
                
                # Calculate spectrograms for Pz
                target_pz = avg_target[pz_idx]
                non_target_pz = avg_non_target[pz_idx]
                
                # Create spectrograms
                fs = session_data.get('sampling_rate', 256)
                
                plt.figure(figsize=(12, 8))
                
                # Target spectrogram
                plt.subplot(2, 1, 1)
                f, t, Sxx = signal.spectrogram(target_pz, fs=fs, nperseg=32, noverlap=16)
                plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
                plt.colorbar(label='Power/Frequency (dB/Hz)')
                plt.title('Target Stimulus - Pz Spectrogram')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.ylim(0, 30)  # Limit to lower frequencies where P300 components exist
                
                # Non-target spectrogram
                plt.subplot(2, 1, 2)
                f, t, Sxx = signal.spectrogram(non_target_pz, fs=fs, nperseg=32, noverlap=16)
                plt.pcolormesh(t, f, 10 * np.log10(Sxx), shading='gouraud')
                plt.colorbar(label='Power/Frequency (dB/Hz)')
                plt.title('Non-target Stimulus - Pz Spectrogram')
                plt.xlabel('Time (s)')
                plt.ylabel('Frequency (Hz)')
                plt.ylim(0, 30)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'time_freq_{file_basename}.png'), dpi=300)
                plt.close()
            
            return True
    except Exception as e:
        print(f"Error extracting ERP plots: {e}")
        return False

def extract_online_performance(session_files):
    """Extract and plot online performance metrics across sessions"""
    try:
        # Collect data from all speller session files
        speller_sessions = [f for f in session_files if 'speller_session' in f]
        
        if not speller_sessions:
            return False
            
        accuracies = []
        itrs = []  # Information Transfer Rates
        session_names = []
        
        for session_file in speller_sessions:
            session_data = load_pkl_file(session_file)
            basename = os.path.basename(session_file).replace('.pkl', '')
            session_names.append(basename)
            
            if 'accuracy' in session_data:
                accuracies.append(session_data['accuracy'] * 100)  # Convert to percentage
            else:
                accuracies.append(0)
                
            if 'itr' in session_data:
                itrs.append(session_data['itr'])
            else:
                itrs.append(0)
        
        # Create bar chart of accuracies
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(session_names))
        width = 0.35
        
        plt.bar(x, accuracies, width, label='Accuracy (%)', color='steelblue')
        
        # Add a horizontal line for average accuracy
        avg_accuracy = np.mean(accuracies)
        plt.axhline(y=avg_accuracy, color='r', linestyle='--', 
                   label=f'Avg Accuracy: {avg_accuracy:.1f}%')
        
        plt.xlabel('Session')
        plt.ylabel('Accuracy (%)')
        plt.title('BCI Spelling Accuracy Across Sessions')
        plt.xticks(x, [s[-6:] for s in session_names], rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spelling_accuracy.png'), dpi=300)
        plt.close()
        
        # Create bar chart of ITRs
        plt.figure(figsize=(10, 6))
        
        plt.bar(x, itrs, width, label='ITR (bits/min)', color='forestgreen')
        
        # Add a horizontal line for average ITR
        avg_itr = np.mean(itrs)
        plt.axhline(y=avg_itr, color='r', linestyle='--', 
                   label=f'Avg ITR: {avg_itr:.1f} bits/min')
        
        plt.xlabel('Session')
        plt.ylabel('Information Transfer Rate (bits/min)')
        plt.title('BCI Information Transfer Rate Across Sessions')
        plt.xticks(x, [s[-6:] for s in session_names], rotation=45)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'spelling_itr.png'), dpi=300)
        plt.close()
        
        return True
    except Exception as e:
        print(f"Error extracting online performance: {e}")
        return False

def main():
    """Main function to extract plots for LaTeX report"""
    # Get all PKL files
    pkl_files = get_all_pkl_files()
    
    if not pkl_files:
        print("No PKL files found in the data directory.")
        return
    
    print(f"Found {len(pkl_files)} PKL files. Processing...")
    
    # Find model file
    model_file = None
    for file in pkl_files:
        if 'p300_model' in file:
            model_file = file
            break
    
    if model_file:
        print(f"Extracting metrics from model file: {os.path.basename(model_file)}")
        extract_model_metrics(model_file)
    
    # Process oddball session files for ERP plots
    oddball_sessions = [f for f in pkl_files if 'oddball_session' in f]
    if oddball_sessions:
        print(f"Processing {len(oddball_sessions)} oddball session files for ERP plots...")
        for session_file in oddball_sessions:
            print(f"  - Extracting ERPs from: {os.path.basename(session_file)}")
            extract_erp_plots(session_file)
    
    # Extract online performance metrics
    print("Extracting online performance metrics...")
    extract_online_performance(pkl_files)
    
    print(f"All figures saved to: {output_dir}")

if __name__ == "__main__":
    main()