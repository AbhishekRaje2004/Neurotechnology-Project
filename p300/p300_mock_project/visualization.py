import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import pickle
import os
import threading
import time
from scipy import signal
import queue

class LiveEEGPlotter:
    """Real-time EEG signal visualization during experiments"""
    def __init__(self):
        self.data_queue = queue.Queue()
        self.markers_queue = queue.Queue()
        self.is_running = False
        self.window = None
        self.max_points = 1000  # Maximum number of points to display
        self.raw_eeg_data = []
        self.timestamps = []
        self.markers = []
        self.update_interval = 100  # ms between updates
        
    def initialize(self, window_title="Live EEG Signal"):
        """Initialize the visualization window"""
        self.window = tk.Toplevel()
        self.window.title(window_title)
        self.window.geometry("900x600")
        
        # Create the figure for plotting
        self.fig = plt.figure(figsize=(9, 6), dpi=100)
        self.ax1 = self.fig.add_subplot(211)  # EEG signal
        self.ax2 = self.fig.add_subplot(212)  # Spectrogram
        
        # Set up titles and labels
        self.ax1.set_title("Real-time EEG Signal")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude (µV)")
        self.ax1.grid(True)
        
        self.ax2.set_title("EEG Spectrogram")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Frequency (Hz)")
        
        # Create a canvas for the plot
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Add control buttons
        controls_frame = ttk.Frame(self.window)
        controls_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(controls_frame, text="Start", command=self.start).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Stop", command=self.stop).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Clear", command=self.clear).pack(side=tk.LEFT, padx=5)
        ttk.Button(controls_frame, text="Save", command=self.save_data).pack(side=tk.RIGHT, padx=5)
        
        # Protocol to handle window close event
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Initialize plot data
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=1)
        self.spectrogram = None
        
        # Start the updating using Tkinter's after method (thread safe)
        self.is_running = True
        self.window.after(self.update_interval, self._update_plot)
        
        return self.window
    
    def add_data_point(self, timestamp, eeg_value, marker=None):
        """Add a new EEG data point to the visualization"""
        self.data_queue.put((timestamp, eeg_value))
        if marker is not None:
            self.markers_queue.put((timestamp, marker))
    
    def _update_plot(self):
        """Update the plot with new data (runs in Tkinter's main thread)"""
        if not self.is_running:
            return
            
        try:
            # Process all available data points
            data_count = 0
            while not self.data_queue.empty():
                timestamp, eeg_value = self.data_queue.get_nowait()
                self.timestamps.append(timestamp)
                self.raw_eeg_data.append(eeg_value)
                data_count += 1
            
            # Process all available markers
            marker_count = 0
            while not self.markers_queue.empty():
                timestamp, marker = self.markers_queue.get_nowait()
                self.markers.append((timestamp, marker))
                marker_count += 1
            
            # Print diagnostic info if data received
            if data_count > 0 or marker_count > 0:
                print(f"Received {data_count} data points, {marker_count} markers. Total points: {len(self.timestamps)}")
            
            # Only keep the most recent data points
            if len(self.timestamps) > self.max_points:
                self.timestamps = self.timestamps[-self.max_points:]
                self.raw_eeg_data = self.raw_eeg_data[-self.max_points:]
            
            if len(self.timestamps) > 10:  # Need some minimum number of points to plot
                # Update signal plot
                relative_time = [t - self.timestamps[0] for t in self.timestamps]
                self.line1.set_data(relative_time, self.raw_eeg_data)
                
                # Adjust limits with some padding
                y_min = min(self.raw_eeg_data) if self.raw_eeg_data else 0
                y_max = max(self.raw_eeg_data) if self.raw_eeg_data else 1
                y_range = max(y_max - y_min, 1)  # Avoid zero range
                
                self.ax1.set_xlim(min(relative_time), max(relative_time))
                self.ax1.set_ylim(y_min - 0.1*y_range, y_max + 0.1*y_range)
                
                # Clear previous marker lines
                for line in self.ax1.lines[1:]:
                    line.remove()
                
                # Plot markers as vertical lines
                for marker_ts, marker_code in self.markers:
                    if marker_ts >= self.timestamps[0] and marker_ts <= self.timestamps[-1]:
                        rel_ts = marker_ts - self.timestamps[0]
                        if marker_code == 1:  # Target marker
                            self.ax1.axvline(rel_ts, color='r', alpha=0.5, linestyle='--')
                        else:
                            self.ax1.axvline(rel_ts, color='g', alpha=0.3, linestyle=':')
                
                # Update spectrogram (when we have enough data)
                if len(self.raw_eeg_data) > 100:
                    self.ax2.clear()
                    try:
                        # Convert lists to numpy arrays to fix the shape error
                        eeg_array = np.array(self.raw_eeg_data)
                        timestamps_array = np.array(self.timestamps)
                        
                        # Calculate approximate sampling rate
                        if len(timestamps_array) > 1:
                            fs = len(timestamps_array) / (timestamps_array[-1] - timestamps_array[0])
                            # Apply spectrogram to the numpy array
                            f, t, Sxx = signal.spectrogram(eeg_array, fs=fs, 
                                                         nperseg=min(256, len(eeg_array)//2))
                            self.ax2.pcolormesh(t, f, 10 * np.log10(Sxx+1e-10), shading='gouraud')
                            self.ax2.set_ylim(0, 50)  # Limit to 0-50 Hz
                            self.ax2.set_title("EEG Spectrogram")
                            self.ax2.set_xlabel("Time (s)")
                            self.ax2.set_ylabel("Frequency (Hz)")
                    except Exception as e:
                        print(f"Spectrogram error: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Update the canvas
                try:
                    self.canvas.draw_idle()
                except Exception as e:
                    print(f"Canvas draw error: {e}")
        
        except Exception as e:
            print(f"Error in plot update: {e}")
            import traceback
            traceback.print_exc()
            
        # Schedule next update using Tkinter's event loop (thread-safe)
        if self.is_running and self.window:
            self.window.after(self.update_interval, self._update_plot)
    
    def start(self):
        """Start or resume visualization"""
        if not self.is_running:
            self.is_running = True
            # Schedule the update function in Tkinter's event loop
            self.window.after(self.update_interval, self._update_plot)
    
    def stop(self):
        """Pause visualization"""
        self.is_running = False
        
    def clear(self):
        """Clear all data"""
        self.timestamps = []
        self.raw_eeg_data = []
        self.markers = []
        self.ax1.clear()
        self.ax2.clear()
        self.ax1.set_title("Real-time EEG Signal")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude (µV)")
        self.ax1.grid(True)
        self.ax2.set_title("EEG Spectrogram")
        self.ax2.set_xlabel("Time (s)")
        self.ax2.set_ylabel("Frequency (Hz)")
        self.line1, = self.ax1.plot([], [], 'b-', linewidth=1)
        self.canvas.draw()
    
    def save_data(self):
        """Save current data to a file"""
        if len(self.timestamps) == 0:
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            data = {
                'timestamps': self.timestamps,
                'eeg_data': self.raw_eeg_data,
                'markers': self.markers
            }
            
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
                
    def on_close(self):
        """Handle window close event"""
        # First stop any running processes
        self.is_running = False
        
        # Flush queues to prevent any blockages
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except:
                pass
                
        while not self.markers_queue.empty():
            try:
                self.markers_queue.get_nowait()
            except:
                pass
        
        # Destroy the window
        if self.window:
            self.window.destroy()
            self.window = None
            
        # Force matplotlib to close all figures associated with this instance
        plt.close(self.fig)


class EpochViewer:
    """Post-experiment epoch visualization and analysis"""
    def __init__(self):
        self.root = None
        self.epochs = None
        self.labels = None
        self.current_epoch_idx = 0
        self.current_view = "single"  # "single" or "average"
        
    def show(self, epochs=None, labels=None, fs=250, pre=0.2, post=0.8):
        """Show the epoch viewer window"""
        # Create root window if it doesn't exist
        if self.root is None:
            self.root = tk.Toplevel()
            self.root.title("P300 Epoch Viewer")
            self.root.geometry("1000x800")
            self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
        # Create UI framework
        self.create_ui(fs, pre, post)
        
        # Load data if provided
        if epochs is not None and labels is not None:
            self.load_data(epochs, labels, fs, pre, post)
        
        return self.root
    
    def create_ui(self, fs, pre, post):
        """Create the user interface"""
        # Main layout frames
        control_frame = ttk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=10, padx=10)
        
        view_control_frame = ttk.Frame(self.root)
        view_control_frame.pack(side=tk.TOP, fill=tk.X, pady=5, padx=10)
        
        self.fig = plt.figure(figsize=(10, 8), dpi=100)
        
        # Signal plot
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_title("EEG Epoch")
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude (µV)")
        self.ax1.grid(True)
        
        # Frequency domain plot
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_title("Frequency Domain")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Power")
        self.ax2.grid(True)
        
        # Add canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self.root)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Add control buttons
        ttk.Button(control_frame, text="Load Epochs", command=self.load_epochs_dialog).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Previous", command=self.prev_epoch).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next", command=self.next_epoch).pack(side=tk.LEFT, padx=5)
        self.epoch_label = ttk.Label(control_frame, text="Epoch: 0/0")
        self.epoch_label.pack(side=tk.LEFT, padx=20)
        self.class_label = ttk.Label(control_frame, text="Class: N/A")
        self.class_label.pack(side=tk.LEFT, padx=20)
        
        # View control
        ttk.Label(view_control_frame, text="View:").pack(side=tk.LEFT, padx=5)
        self.view_var = tk.StringVar(value="single")
        ttk.Radiobutton(view_control_frame, text="Single Epoch", variable=self.view_var, value="single", 
                        command=self.update_view).pack(side=tk.LEFT, padx=5)
        ttk.Radiobutton(view_control_frame, text="Average", variable=self.view_var, value="average", 
                        command=self.update_view).pack(side=tk.LEFT, padx=5)
        
        # Filter options
        ttk.Label(view_control_frame, text="Filter:").pack(side=tk.LEFT, padx=20)
        self.filter_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(view_control_frame, text="Apply Filters", variable=self.filter_var, 
                       command=self.update_plot).pack(side=tk.LEFT, padx=5)
    
    def load_epochs_dialog(self):
        """Open file dialog to load epoch data"""
        filename = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if not filename:
            return
            
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                
            if 'epochs' in data and 'labels' in data:
                self.load_data(data['epochs'], data['labels'], 
                              data.get('fs', 250), 
                              data.get('pre', 0.2), 
                              data.get('post', 0.8))
            else:
                print("Invalid epoch data format")
        except Exception as e:
            print(f"Error loading file: {e}")
    
    def load_data(self, epochs, labels, fs=250, pre=0.2, post=0.8):
        """Load epoch data for visualization"""
        self.epochs = np.array(epochs)
        self.labels = np.array(labels)
        self.fs = fs
        self.pre = pre
        self.post = post
        self.current_epoch_idx = 0
        
        # Create time vector
        self.time_vector = np.linspace(-pre, post, self.epochs.shape[2])
        
        # Update the plot with the loaded data
        self.update_plot()
    
    def update_plot(self):
        """Update the plot with current epoch data"""
        if self.epochs is None or len(self.epochs) == 0:
            return
        
        # Clear the axes
        self.ax1.clear()
        self.ax2.clear()
        
        # Update the time-domain plot
        if self.view_var.get() == "single":
            # Single epoch view
            epoch_data = self.epochs[self.current_epoch_idx, 0, :]
            if self.filter_var.get():
                # Apply basic filtering
                from scipy import signal
                b, a = signal.butter(4, [1, 20], fs=self.fs, btype='band')
                epoch_data = signal.filtfilt(b, a, epoch_data)
                
            self.ax1.plot(self.time_vector, epoch_data, 'b-', linewidth=1)
            self.ax1.set_title(f"Epoch {self.current_epoch_idx+1}/{len(self.epochs)} " + 
                               f"({'Target' if self.labels[self.current_epoch_idx] == 1 else 'Non-target'})")
            
        else:
            # Average view
            target_idx = [i for i, l in enumerate(self.labels) if l == 1]
            non_target_idx = [i for i, l in enumerate(self.labels) if l == 0]
            
            if target_idx:
                target_avg = np.mean(self.epochs[target_idx, 0, :], axis=0)
                if self.filter_var.get():
                    from scipy import signal
                    b, a = signal.butter(4, [1, 20], fs=self.fs, btype='band')
                    target_avg = signal.filtfilt(b, a, target_avg)
                self.ax1.plot(self.time_vector, target_avg, 'b-', linewidth=2, label='Target (P300)')
            
            if non_target_idx:
                non_target_avg = np.mean(self.epochs[non_target_idx, 0, :], axis=0)
                if self.filter_var.get():
                    from scipy import signal
                    b, a = signal.butter(4, [1, 20], fs=self.fs, btype='band')
                    non_target_avg = signal.filtfilt(b, a, non_target_avg)
                self.ax1.plot(self.time_vector, non_target_avg, 'r-', linewidth=2, label='Non-target')
            
            self.ax1.legend()
            self.ax1.set_title(f"Average ERPs (Targets: {len(target_idx)}, Non-targets: {len(non_target_idx)})")
        
        # Plot trigger line
        self.ax1.axvline(0, color='k', linestyle='--', label='Stimulus')
        
        # Highlight P300 region
        self.ax1.axvspan(0.25, 0.5, color='yellow', alpha=0.2, label='P300 window')
        
        # Set labels and grid
        self.ax1.set_xlabel("Time (s)")
        self.ax1.set_ylabel("Amplitude (µV)")
        self.ax1.grid(True)
        
        # Update the frequency domain plot
        if self.view_var.get() == "single":
            epoch_data = self.epochs[self.current_epoch_idx, 0, :]
            if self.filter_var.get():
                from scipy import signal
                b, a = signal.butter(4, [1, 20], fs=self.fs, btype='band')
                epoch_data = signal.filtfilt(b, a, epoch_data)
                
            # Compute power spectral density
            f, pxx = signal.welch(epoch_data, fs=self.fs, nperseg=min(256, len(epoch_data)))
            self.ax2.plot(f, pxx)
            self.ax2.set_xlim(0, 40)  # Limit to 0-40 Hz
            
        else:
            # Plot average spectra
            target_idx = [i for i, l in enumerate(self.labels) if l == 1]
            non_target_idx = [i for i, l in enumerate(self.labels) if l == 0]
            
            if target_idx:
                target_avg = np.mean(self.epochs[target_idx, 0, :], axis=0)
                if self.filter_var.get():
                    from scipy import signal
                    b, a = signal.butter(4, [1, 20], fs=self.fs, btype='band')
                    target_avg = signal.filtfilt(b, a, target_avg)
                f, pxx = signal.welch(target_avg, fs=self.fs, nperseg=min(256, len(target_avg)))
                self.ax2.plot(f, pxx, 'b-', label='Target (P300)')
            
            if non_target_idx:
                non_target_avg = np.mean(self.epochs[non_target_idx, 0, :], axis=0)
                if self.filter_var.get():
                    from scipy import signal
                    b, a = signal.butter(4, [1, 20], fs=self.fs, btype='band')
                    non_target_avg = signal.filtfilt(b, a, non_target_avg)
                f, pxx = signal.welch(non_target_avg, fs=self.fs, nperseg=min(256, len(non_target_avg)))
                self.ax2.plot(f, pxx, 'r-', label='Non-target')
                
            self.ax2.legend()
            
        self.ax2.set_title("Power Spectral Density")
        self.ax2.set_xlabel("Frequency (Hz)")
        self.ax2.set_ylabel("Power (µV²/Hz)")
        self.ax2.grid(True)
        self.ax2.set_xlim(0, 40)
        
        # Update info labels
        self.epoch_label.config(text=f"Epoch: {self.current_epoch_idx+1}/{len(self.epochs)}")
        self.class_label.config(text=f"Class: {'Target' if self.labels[self.current_epoch_idx] == 1 else 'Non-target'}")
        
        # Update canvas
        self.canvas.draw()
    
    def next_epoch(self):
        """Go to next epoch"""
        if self.epochs is None or len(self.epochs) == 0:
            return
            
        self.current_epoch_idx = (self.current_epoch_idx + 1) % len(self.epochs)
        self.update_plot()
    
    def prev_epoch(self):
        """Go to previous epoch"""
        if self.epochs is None or len(self.epochs) == 0:
            return
            
        self.current_epoch_idx = (self.current_epoch_idx - 1) % len(self.epochs)
        self.update_plot()
    
    def update_view(self):
        """Switch between single epoch and average view"""
        self.current_view = self.view_var.get()
        self.update_plot()
    
    def on_close(self):
        """Handle window close event"""
        # Make sure to close the matplotlib figure to free resources
        plt.close(self.fig)
        
        # Destroy the window
        if self.root:
            self.root.destroy()
            self.root = None


def save_experiment_data(epochs, labels, timestamps=None, eeg_data=None, events=None, fs=250, pre=0.2, post=0.8, filename=None):
    """Save experiment data to a file for later analysis"""
    if filename is None:
        filename = f"p300_experiment_data_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        
    data = {
        'epochs': epochs,
        'labels': labels,
        'fs': fs,
        'pre': pre,
        'post': post,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add raw data if available
    if timestamps is not None:
        data['timestamps'] = timestamps
    if eeg_data is not None:
        data['eeg_data'] = eeg_data
    if events is not None:
        data['events'] = events
    
    with open(filename, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"Experiment data saved to {filename}")
    return filename


def load_experiment_data(filename):
    """Load saved experiment data"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

def test_live_visualization(port='COM4', duration=30, gain=1.0):
    """Run a standalone test of the live visualization with Arduino data
    
    Parameters:
    -----------
    port : str
        Serial port for Arduino
    duration : int
        Duration in seconds to run the test
    gain : float
        Signal amplification factor for better visualization
    """
    import serial
    import time
    import tkinter as tk
    
    print(f"Starting visualization test on port {port} with gain {gain}...")
    
    # Create Tkinter root
    root = tk.Tk()
    root.title("Arduino Signal Test")
    root.geometry("200x100")
    
    # Add a status label
    status_label = tk.Label(root, text="Connecting to Arduino...")
    status_label.pack(pady=20)
    
    # Create visualization
    plotter = LiveEEGPlotter()
    viz_window = plotter.initialize("Signal Visualization Test")
    
    # Try to connect to Arduino
    ser = None
    try:
        ser = serial.Serial(port, 115200, timeout=1)
        status_label.config(text=f"Connected to {port}")
        print(f"Connected to {port}")
        time.sleep(2)  # Wait for Arduino to reset
        
        # Clear any pending data
        ser.reset_input_buffer()
        
        # Set up sample counter
        sample_count = [0]  # Using a list for nonlocal access
        start_time = time.time()
        end_time = start_time + duration
        
        def read_arduino_data():
            if time.time() < end_time and ser and ser.is_open:
                if ser.in_waiting > 0:
                    try:
                        line = ser.readline().decode('utf-8').strip()
                        parts = line.split(',')
                        if len(parts) == 3:
                            # Extract data from CSV format
                            arduino_ts = int(parts[0])
                            eeg_value = float(parts[1]) * gain  # Apply gain
                            trigger = int(parts[2])
                            
                            # Add to visualization
                            plotter.add_data_point(time.time(), eeg_value, 
                                                  trigger if trigger >= 0 else None)
                            sample_count[0] += 1
                            
                            # Print some feedback
                            if sample_count[0] % 100 == 0:
                                status_label.config(text=f"Samples: {sample_count[0]}")
                                print(f"Received {sample_count[0]} samples, latest value: {eeg_value}")
                    except Exception as e:
                        print(f"Error processing data: {e}")
                
                # Schedule the next reading
                root.after(5, read_arduino_data)
            else:
                # Test duration completed
                status_label.config(text=f"Complete: {sample_count[0]} samples")
                print(f"Test complete. Collected {sample_count[0]} samples in {duration} seconds.")
                
                # Close the Arduino connection
                if ser and ser.is_open:
                    ser.close()
                    print("Serial connection closed")
        
        # Start reading data (using Tkinter's event scheduling)
        root.after(10, read_arduino_data)
        
        # Run the mainloop
        root.mainloop()
        
    except Exception as e:
        status_label.config(text=f"Error: {str(e)}")
        print(f"Error connecting to Arduino: {e}")
        if ser and ser.is_open:
            ser.close()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        port = sys.argv[1]
        gain = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
        test_live_visualization(port=port, gain=gain)
    else:
        test_live_visualization()