import serial
import time
import numpy as np
import threading
from scipy import signal
import collections

class ArduinoEEGAcquisition:
    def __init__(self, port='COM4', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.ser = None
        self.is_recording = False
        self.data_thread = None
        self.eeg_data = []
        self.timestamps = []
        self.triggers = []
        
        # For sampling rate calculation
        self.expected_fs = 250  # Expected sampling rate
        self.sample_count = 0
        self.start_time = 0
        self.last_report_time = 0
        self.report_interval = 5  # Report sampling rate every 5 seconds
        
        # For real-time 50Hz filtering
        self.filter_queue = collections.deque(maxlen=25)  # Buffer for filtering (~100ms at 250Hz)
        self.apply_filters = True
        
    def connect(self):
        """Connect to Arduino"""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
            # Wait for Arduino to initialize
            time.sleep(2)
            
            # Wait for initialization message
            init_received = False
            init_timeout = time.time() + 5  # 5 second timeout
            
            while time.time() < init_timeout and not init_received:
                if self.ser.in_waiting > 0:
                    try:
                        line = self.ser.readline().decode('utf-8').strip()
                        if line.startswith("EEG,Arduino_Init"):
                            print(f"Arduino initialized: {line}")
                            init_received = True
                    except:
                        pass
                    
            if not init_received:
                print("Warning: No initialization message received from Arduino")
                
            return True
        except Exception as e:
            print(f"Connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.ser and self.ser.is_open:
            self.ser.close()
            
    def start_acquisition(self, apply_filters=True):
        """Start continuous data acquisition in a separate thread
        
        Parameters:
        -----------
        apply_filters : bool
            Whether to apply real-time filters (including 50Hz notch filter)
        """
        if self.ser is None or not self.ser.is_open:
            if not self.connect():
                return False
        
        self.apply_filters = apply_filters
        
        # Start acquisition thread
        self.is_recording = True
        self.data_thread = threading.Thread(target=self._acquisition_loop)
        self.data_thread.daemon = True
        self.data_thread.start()
        return True
    
    def stop_acquisition(self):
        """Stop continuous data acquisition"""
        self.is_recording = False
        if self.data_thread:
            self.data_thread.join(timeout=1.0)
            self.data_thread = None
    
    def _acquisition_loop(self):
        """Main acquisition loop (runs in a separate thread)"""
        self.eeg_data = []
        self.timestamps = []
        self.triggers = []
        
        # Reset counters for sampling rate calculation
        self.sample_count = 0
        self.start_time = time.time()
        self.last_report_time = self.start_time
        
        # Flush any pending data
        self.ser.reset_input_buffer()
        
        # Initialize filters
        if self.apply_filters:
            # Create 50Hz notch filter
            b_notch, a_notch = signal.iirnotch(50, 30, self.expected_fs)
            
            # Create bandpass filter
            b_band, a_band = signal.butter(4, [1, 40], btype='band', fs=self.expected_fs)
            
            # Initialize filter states
            self.z_notch = signal.lfilter_zi(b_notch, a_notch)
            self.z_band = signal.lfilter_zi(b_band, a_band)
        
        # Read until stopped
        while self.is_recording:
            try:
                if self.ser.in_waiting > 0:
                    line = self.ser.readline().decode('utf-8').strip()
                    parts = line.split(',')
                    
                    if len(parts) == 3:
                        arduino_ts = int(parts[0])  # Arduino timestamp (ms)
                        eeg_value = float(parts[1])  # EEG value
                        trigger = int(parts[2])      # Trigger value (-1 if none)
                        
                        # Apply real-time filtering if enabled
                        if self.apply_filters:
                            # Add to filter queue for processing
                            self.filter_queue.append(eeg_value)
                            
                            if len(self.filter_queue) == self.filter_queue.maxlen:
                                # Apply filters to the buffer
                                filtered, self.z_notch = signal.lfilter(b_notch, a_notch, 
                                                                     list(self.filter_queue), 
                                                                     zi=self.z_notch)
                                filtered, self.z_band = signal.lfilter(b_band, a_band, 
                                                                    filtered, 
                                                                    zi=self.z_band)
                                
                                # Use the filtered value
                                eeg_value = filtered[-1]
                        
                        # Record the data
                        self.eeg_data.append([eeg_value])
                        current_time = time.time()
                        self.timestamps.append(current_time)
                        self.triggers.append(trigger if trigger >= 0 else None)
                        
                        # Increment sample counter
                        self.sample_count += 1
                        
                        # Check and report sampling rate periodically
                        if current_time - self.last_report_time >= self.report_interval:
                            elapsed = current_time - self.start_time
                            actual_fs = self.sample_count / elapsed if elapsed > 0 else 0
                            
                            # Report if sampling rate deviates more than 5% from expected
                            if abs(actual_fs - self.expected_fs) > (self.expected_fs * 0.05):
                                print(f"Warning: Sampling rate is {actual_fs:.1f}Hz (expected {self.expected_fs}Hz)")
                            else:
                                print(f"Sampling rate OK: {actual_fs:.1f}Hz")
                                
                            self.last_report_time = current_time
                            
            except Exception as e:
                print(f"Error reading data: {e}")
                time.sleep(0.01)  # Short delay to prevent CPU overload
                continue
        
    def get_data(self):
        """Get the currently acquired data"""
        return self.timestamps.copy(), self.eeg_data.copy(), self.triggers.copy()
    
    def get_sampling_rate(self):
        """Calculate actual sampling rate"""
        if self.sample_count > 0 and self.start_time > 0:
            elapsed = time.time() - self.start_time
            if elapsed > 0:
                return self.sample_count / elapsed
        return 0
    
    def clear_data(self):
        """Clear the stored data"""
        self.eeg_data = []
        self.timestamps = []
        self.triggers = []
        self.sample_count = 0
        self.start_time = time.time()

def read_serial_eeg(port='COM4', baudrate=115200, duration=60, apply_filters=True):
    """Read EEG data from Arduino for a specified duration
    
    Parameters:
    -----------
    port : str
        Serial port for Arduino
    baudrate : int
        Baud rate for serial connection
    duration : float
        Recording duration in seconds
    apply_filters : bool
        Whether to apply real-time filters (including 50Hz notch)
    
    Returns:
    --------
    timestamps : list
        List of timestamps
    eeg_data : list
        List of EEG values
    triggers : list
        List of trigger values
    """
    acquisition = ArduinoEEGAcquisition(port, baudrate)
    if not acquisition.connect():
        return [], [], []
        
    acquisition.start_acquisition(apply_filters)
    
    print(f"Recording for {duration} seconds...")
    time.sleep(duration)  # Record for specified duration
    
    acquisition.stop_acquisition()
    
    timestamps, eeg_data, triggers = acquisition.get_data()
    acquisition.disconnect()
    
    # Report final sampling rate
    actual_fs = len(timestamps) / duration if duration > 0 else 0
    print(f"Acquisition complete: {len(timestamps)} samples collected")
    print(f"Average sampling rate: {actual_fs:.1f}Hz")
    
    return timestamps, eeg_data, triggers

if __name__ == "__main__":
    import sys
    import matplotlib.pyplot as plt
    from preprocessing import remove_line_noise
    
    # Default parameters
    port = 'COM4'
    duration = 10
    apply_filters = True
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        port = sys.argv[1]
    if len(sys.argv) > 2:
        duration = float(sys.argv[2])
    if len(sys.argv) > 3:
        apply_filters = sys.argv[3].lower() != 'false'
    
    # Run acquisition
    print(f"Starting acquisition from {port} for {duration}s (filters: {apply_filters})")
    timestamps, eeg_data, triggers = read_serial_eeg(port, 115200, duration, apply_filters)
    
    # If data was collected, plot it
    if len(timestamps) > 0:
        # Convert to numpy array for easier processing
        eeg_array = np.array([data[0] for data in eeg_data])
        ts_array = np.array(timestamps)
        relative_time = ts_array - ts_array[0]
        
        # Calculate sampling rate
        fs = len(timestamps) / (timestamps[-1] - timestamps[0]) if len(timestamps) > 1 else 0
        print(f"Calculated sampling rate: {fs:.1f}Hz")
        
        # Create plot
        plt.figure(figsize=(12, 8))
        
        # Plot raw signal
        plt.subplot(2, 1, 1)
        plt.plot(relative_time, eeg_array, 'b-', linewidth=1, alpha=0.7, label='Raw')
        
        # Add 50Hz notch filtered signal for comparison
        if len(eeg_array) > 50:  # Need enough data for filtering
            filtered = remove_line_noise(eeg_array, fs=fs)
            plt.plot(relative_time, filtered, 'r-', linewidth=1, label='50Hz Filtered')
            
        plt.title('EEG Signal')
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.grid(True)
        plt.legend()
        
        # Plot frequency spectrum
        plt.subplot(2, 1, 2)
        if len(eeg_array) > 100:  # Need enough data for spectrum
            f, Pxx = signal.welch(eeg_array, fs=fs, nperseg=min(256, len(eeg_array)//2))
            plt.semilogy(f, Pxx, 'b-', alpha=0.7, label='Raw')
            
            # Add filtered spectrum
            f_filt, Pxx_filt = signal.welch(filtered, fs=fs, nperseg=min(256, len(filtered)//2))
            plt.semilogy(f_filt, Pxx_filt, 'r-', label='Filtered')
            
            plt.title('Power Spectrum')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density')
            plt.grid(True)
            plt.axvline(50, color='gray', linestyle='--', alpha=0.7, label='50 Hz')
            plt.legend()
            plt.xlim([0, fs/2])  # Limit to Nyquist frequency
        
        plt.tight_layout()
        plt.show()
    else:
        print("No data collected!")
