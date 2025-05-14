import serial
import csv
import os
import statistics
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from serial.tools import list_ports
import shutil

# === Configuration ===
BAUD_RATE = 115200
EXPECTED_PERIOD_MS = 800.0  # Expected pulse period in milliseconds
DEFAULT_DURATION_SECONDS = 60  # Default duration if user doesn't specify

# ESP32 ADC Configuration
ADC_RESOLUTION = 12  # 12-bit ADC
ADC_MAX_VALUE = (2**ADC_RESOLUTION) - 1  # 4095 for 12-bit
ADC_VOLTAGE_REFERENCE = 3.3  # ESP32 voltage reference is 3.3V
VOLTAGE_MULTIPLIER = 2.0  # Actual voltage is twice the measured value

def adc_to_voltage(adc_value):
    """Convert ADC reading to voltage value, applying the multiplier"""
    raw_voltage = (adc_value / ADC_MAX_VALUE) * ADC_VOLTAGE_REFERENCE
    return raw_voltage * VOLTAGE_MULTIPLIER

def get_session_folder_name():
    """Get a name for the session folder"""
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    # Update to include the pulse_ prefix for the folder name
    default_name = f"pulse_session_{timestamp_str}"
    
    print("\nSession Folder Name")
    print("------------------")
    folder_name = input(f"Enter a name for this recording session (default: '{default_name}'): ")
    
    if not folder_name:
        return default_name
    
    # Remove invalid characters for folders
    folder_name = "".join([c for c in folder_name if c.isalnum() or c in " _-"])
    
    # If user provided a custom name without prefix, add the prefix
    if not folder_name.startswith(('power_', 'pulse_', 'waveform_')):
        folder_name = f"pulse_{folder_name}"
        
    return folder_name

def find_available_ports():
    """Auto-detect available serial ports and return a list of them"""
    available_ports = []
    ports = list(list_ports.comports())
    for port in ports:
        available_ports.append(port.device)
    return available_ports

def select_port():
    """Provide a user interface to select an available serial port"""
    ports = find_available_ports()
    
    if not ports:
        print("No serial ports found. Please check connections and try again.")
        return None
        
    print("\nAvailable Serial Ports:")
    print("------------------------")
    for i, port in enumerate(ports):
        print(f"[{i+1}] {port}")
    
    while True:
        try:
            choice = input("\nSelect port number (or press Enter for auto-connect): ")
            if choice == "":
                # Auto-connect to the first available port
                print(f"Auto-connecting to {ports[0]}...")
                return ports[0]
            elif choice.isdigit() and 1 <= int(choice) <= len(ports):
                selected_port = ports[int(choice)-1]
                return selected_port
            else:
                print("Invalid selection. Please try again.")
        except Exception as e:
            print(f"Error: {e}")

def get_filename_prefix():
    """Get a custom prefix for output filenames"""
    default_prefix = "pulse"
    prefix = input(f"\nEnter filename prefix (default: '{default_prefix}'): ")
    if not prefix:
        return default_prefix
    return prefix

def get_duration():
    """Allow the user to specify how long the data collection should run"""
    print("\nData Collection Duration")
    print("------------------------")
    print("[1] 30 seconds")
    print("[2] 60 seconds (default)")
    print("[3] 2 minutes")
    print("[4] 5 minutes")
    print("[5] Custom duration")
    
    while True:
        try:
            choice = input("\nSelect duration or press Enter for default (60s): ")
            
            if choice == "":
                return DEFAULT_DURATION_SECONDS
            elif choice == "1":
                return 30
            elif choice == "2":
                return 60
            elif choice == "3":
                return 120
            elif choice == "4":
                return 300
            elif choice == "5":
                try:
                    custom = input("Enter custom duration in seconds: ")
                    duration = int(custom)
                    if duration <= 0:
                        print("Duration must be positive. Please try again.")
                        continue
                    return duration
                except ValueError:
                    print("Invalid input. Please enter a number.")
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please try again.")

# === Line parser ===
def parse_line(line):
    try:
        parts = line.strip().split(',')
        if len(parts) != 4:
            return None
        adc_value = int(parts[1])
        return {
            'timestamp': int(parts[0]),
            'adc_value': adc_value,
            'voltage': adc_to_voltage(adc_value),  # Convert ADC to actual voltage
            'pulse': int(parts[2]),
            'delta': int(parts[3])
        }
    except ValueError:
        return None

def plot_statistics(data, timestamps, voltages, pulses, delta_times, file_prefix, session_path):
    """Generate separate statistical plots for pulse data - one file per plot for clarity"""
    
    # Create plots directory within the session folder
    plots_dir = os.path.join(session_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_files = []
    
    # === Plot 1: Raw Signal with Detected Pulses ===
    plt.figure(figsize=(10, 6))
    plt.plot(timestamps, voltages, 'b-', linewidth=1, label='Voltage Reading')
    
    # Mark pulse detection points
    pulse_indices = [i for i, pulse in enumerate(pulses) if pulse == 1]
    pulse_timestamps = [timestamps[i] for i in pulse_indices]
    pulse_voltages = [voltages[i] for i in pulse_indices]
    plt.plot(pulse_timestamps, pulse_voltages, 'ro', markersize=5, label='Detected Pulse')
    
    plt.title('Voltage Signal and Detected Pulses', fontsize=14)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the plot
    signal_plot_file = os.path.join(plots_dir, f'{file_prefix}_signal.png')
    plt.savefig(signal_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(signal_plot_file)
    
    # === Plot 2: Pulse Intervals over Time ===
    if pulse_timestamps and len(pulse_timestamps) > 1:
        plt.figure(figsize=(10, 6))
        
        intervals = [pulse_timestamps[i] - pulse_timestamps[i-1] for i in range(1, len(pulse_timestamps))]
        interval_times = [pulse_timestamps[i] for i in range(1, len(pulse_timestamps))]
        
        plt.plot(interval_times, intervals, 'g-o', markersize=5)
        plt.axhline(y=EXPECTED_PERIOD_MS, color='r', linestyle='--', label=f'Expected ({EXPECTED_PERIOD_MS} ms)')
        
        # Add mean line
        mean_interval = statistics.mean(intervals)
        plt.axhline(y=mean_interval, color='blue', linestyle='-', label=f'Mean ({mean_interval:.1f} ms)')
        
        plt.title('Pulse Intervals over Time', fontsize=14)
        plt.xlabel('Time (ms)', fontsize=12)
        plt.ylabel('Interval (ms)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save the plot
        intervals_plot_file = os.path.join(plots_dir, f'{file_prefix}_intervals.png')
        plt.savefig(intervals_plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(intervals_plot_file)
    
    # === Plot 3: Voltage Distribution Histogram ===
    plt.figure(figsize=(10, 6))
    
    # Use seaborn if available for nicer histograms
    try:
        import seaborn as sns
        sns.histplot(voltages, bins=30, kde=True, color='skyblue')
    except ImportError:
        plt.hist(voltages, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add lines for mean and max values
    mean_voltage = statistics.mean(voltages)
    max_voltage = max(voltages)
    plt.axvline(x=mean_voltage, color='g', linestyle='-', linewidth=2, 
                label=f'Mean ({mean_voltage:.3f}V)')
    plt.axvline(x=max_voltage, color='r', linestyle='--', linewidth=2, 
                label=f'Max ({max_voltage:.3f}V)')
    
    plt.title('Voltage Distribution', fontsize=14)
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the plot
    voltage_hist_file = os.path.join(plots_dir, f'{file_prefix}_voltage_hist.png')
    plt.savefig(voltage_hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(voltage_hist_file)
    
    # === Plot 4: Pulse Interval Distribution ===
    non_zero_deltas = [d for d in delta_times if d > 0]
    if non_zero_deltas:
        plt.figure(figsize=(10, 6))
        
        # Use seaborn if available for nicer histograms
        try:
            import seaborn as sns
            sns.histplot(non_zero_deltas, bins=20, kde=True, color='lightgreen')
        except ImportError:
            plt.hist(non_zero_deltas, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        
        # Add vertical line for expected period and mean
        mean_interval = statistics.mean(non_zero_deltas)
        plt.axvline(x=EXPECTED_PERIOD_MS, color='r', linestyle='--', linewidth=2, 
                    label=f'Expected ({EXPECTED_PERIOD_MS} ms)')
        plt.axvline(x=mean_interval, color='g', linestyle='-', linewidth=2, 
                    label=f'Mean ({mean_interval:.1f} ms)')
        
        plt.title('Pulse Interval Distribution', fontsize=14)
        plt.xlabel('Interval (ms)', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        plt.legend(fontsize=12)
        
        # Save the plot
        interval_hist_file = os.path.join(plots_dir, f'{file_prefix}_interval_hist.png')
        plt.savefig(interval_hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        plot_files.append(interval_hist_file)
    
    # Create a summary file listing all plots
    summary_file = os.path.join(session_path, f'{file_prefix}_plots_index.txt')
    with open(summary_file, 'w') as f:
        f.write("PULSE DATA VISUALIZATION FILES\n")
        f.write("=============================\n\n")
        f.write(f"Session: {os.path.basename(session_path)}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for i, plot_file in enumerate(plot_files, 1):
            rel_path = os.path.relpath(plot_file, session_path)
            f.write(f"{i}. {os.path.basename(plot_file)}\n")
            f.write(f"   Path: {rel_path}\n\n")
    
    print(f"\nPlots saved to: {plots_dir}")
    print(f"Generated {len(plot_files)} plot files")
    
    return plots_dir

def save_analysis_to_txt(data, file_prefix, session_path):
    """Save analysis results to a human-readable text file"""
    if not data.get('pulse_intervals'):
        return None
        
    txt_file = os.path.join(session_path, f'{file_prefix}_analysis.txt')
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("PULSE DATA ANALYSIS REPORT\n")
        f.write("==========================\n\n")
        f.write(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data collection session: {os.path.basename(session_path)}\n\n")
        
        f.write("TIMING ANALYSIS\n")
        f.write("--------------\n")
        f.write(f"Total pulses detected: {data['pulse_count']}\n")
        f.write(f"Average pulse period: {data['avg_period']:.2f} ms\n")
        f.write(f"Expected pulse period: {EXPECTED_PERIOD_MS:.2f} ms\n")
        f.write(f"Deviation from expected: {(data['avg_period'] - EXPECTED_PERIOD_MS):.2f} ms\n\n")
        
        f.write("PULSE INTERVAL STATISTICS\n")
        f.write("------------------------\n")
        f.write(f"Minimum interval: {data['min_period']:.2f} ms\n")
        f.write(f"Maximum interval: {data['max_period']:.2f} ms\n")
        f.write(f"Standard deviation: {data['std_dev']:.2f} ms\n")
        
        # Add voltage statistics section
        f.write("\nVOLTAGE STATISTICS\n")
        f.write("------------------\n")
        f.write(f"Mean voltage: {data['mean_voltage']:.3f} V\n")
        f.write(f"Peak voltage: {data['max_voltage']:.3f} V\n") 
        f.write(f"Minimum voltage: {data['min_voltage']:.3f} V\n")
        f.write(f"Voltage range: {data['voltage_range']:.3f} V\n")
        f.write(f"Voltage standard deviation: {data['voltage_std_dev']:.3f} V\n")
        f.write(f"Note: Voltage values reflect actual voltages (2x the measured values)\n\n")
        
        if data.get('pulse_count') > 3:
            f.write("\nINTERVAL STABILITY ANALYSIS\n")
            f.write("-------------------------\n")
            stability = (data['std_dev'] / data['avg_period']) * 100
            f.write(f"Interval stability: {100 - stability:.1f}% (lower std dev = higher stability)\n")
            
            # Calculate frequency in Hz and BPM
            freq_hz = 1000.0 / data['avg_period']
            freq_bpm = freq_hz * 60.0
            f.write(f"Average frequency: {freq_hz:.3f} Hz ({freq_bpm:.1f} BPM)\n")
    
    print(f"Analysis report saved to: {txt_file}")
    return txt_file

# === Main function ===
def main():
    # Main data directory
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    # Auto-connect or select port
    port = select_port()
    if not port:
        print("No port selected. Exiting.")
        return
    
    # Get custom filename prefix
    file_prefix = get_filename_prefix()
    
    # Get session folder name
    session_name = get_session_folder_name()
    session_path = os.path.join(data_dir, session_name)
    os.makedirs(session_path, exist_ok=True)
    
    # Get desired duration
    duration_seconds = get_duration()
    print(f"\nData will be collected for {duration_seconds} seconds.")
    
    # Define output files with custom prefix in session folder
    CSV_FILE_RAW = os.path.join(session_path, f'{file_prefix}_data_raw.csv')
    CSV_FILE_ANALYSIS = os.path.join(session_path, f'{file_prefix}_analysis.csv')
    CSV_FILE_CORRUPTED = os.path.join(session_path, f'{file_prefix}_data_corrupted.csv')
    
    # Storage for data visualization
    timestamps = []
    adc_values = []
    voltages = []
    pulses = []
    delta_times = []
    pulse_intervals = []

    with serial.Serial(port, BAUD_RATE, timeout=1) as ser, \
         open(CSV_FILE_RAW, mode='w', newline='', encoding='utf-8') as file_raw, \
         open(CSV_FILE_CORRUPTED, mode='w', newline='', encoding='utf-8', errors='replace') as file_corrupted:

        writer_raw = csv.writer(file_raw)
        writer_corrupted = csv.writer(file_corrupted)
        writer_raw.writerow(['Timestamp(ms)', 'ADC_Value', 'Voltage(V)', 'PulseDetected', 'TimeSinceLastPulse(ms)'])
        writer_corrupted.writerow(['RawLine'])

        print(f"\nReading serial data from {port} for {duration_seconds} seconds...")
        print("Press Ctrl+C to stop early.")
        
        # Setup timing
        start_time = time.time()
        end_time = start_time + duration_seconds
        last_progress_update = start_time

        try:
            while time.time() < end_time:
                # Show progress every second
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = duration_seconds - elapsed
                
                if current_time - last_progress_update >= 1:
                    progress_pct = int((elapsed / duration_seconds) * 100)
                    print(f"\rProgress: {progress_pct}% - {int(remaining)}s remaining", end='')
                    last_progress_update = current_time
                
                # Check if data is available to read
                if ser.in_waiting > 0:
                    line_bytes = ser.readline()
                    try:
                        line = line_bytes.decode('utf-8', errors='ignore')
                    except UnicodeDecodeError as e:
                        writer_corrupted.writerow([str(line_bytes)])
                        continue

                    data = parse_line(line)
                    if data:
                        writer_raw.writerow([
                            data['timestamp'],
                            data['adc_value'], 
                            data['voltage'],
                            data['pulse'], 
                            data['delta']
                        ])
                        
                        # Store data for plotting
                        timestamps.append(data['timestamp'])
                        adc_values.append(data['adc_value'])
                        voltages.append(data['voltage'])
                        pulses.append(data['pulse'])
                        delta_times.append(data['delta'])
                        
                        if data['pulse'] == 1:
                            pulse_intervals.append(data['delta'])
                    else:
                        writer_corrupted.writerow([line.strip()])
                else:
                    # Sleep briefly to prevent CPU hogging if no data
                    time.sleep(0.01)
                    
            print("\nData collection complete!")
            
        except KeyboardInterrupt:
            elapsed = time.time() - start_time
            print(f"\nData collection stopped by user after {int(elapsed)} seconds.")

    # === Analyze pulse timing and voltage ===
    if voltages and pulse_intervals:
        # Voltage statistics
        mean_voltage = statistics.mean(voltages)
        min_voltage = min(voltages)
        max_voltage = max(voltages) 
        voltage_range = max_voltage - min_voltage
        voltage_std_dev = statistics.stdev(voltages) if len(voltages) > 1 else 0
        
        # Pulse timing statistics
        avg_period = statistics.mean(pulse_intervals)
        std_dev = statistics.stdev(pulse_intervals) if len(pulse_intervals) > 1 else 0
        min_period = min(pulse_intervals)
        max_period = max(pulse_intervals)
        pulse_count = len(pulse_intervals)

        with open(CSV_FILE_ANALYSIS, mode='w', newline='', encoding='utf-8') as file_stats:
            writer = csv.writer(file_stats)
            writer.writerow(['Metric', 'Value', 'Unit', 'Note'])
            
            # Timing metrics
            writer.writerow(['Expected Period', EXPECTED_PERIOD_MS, 'ms', 'Configured'])
            writer.writerow(['Average Period', avg_period, 'ms', 'Measured'])
            writer.writerow(['Min Period', min_period, 'ms', 'Measured'])
            writer.writerow(['Max Period', max_period, 'ms', 'Measured'])
            writer.writerow(['Period Standard Deviation', std_dev, 'ms', 'Jitter'])
            writer.writerow(['Pulse Count', pulse_count, 'pulses', 'Total'])
            
            # Voltage metrics
            writer.writerow(['Mean Voltage', mean_voltage, 'V', 'Includes 2x multiplier'])
            writer.writerow(['Min Voltage', min_voltage, 'V', 'Includes 2x multiplier'])
            writer.writerow(['Max Voltage', max_voltage, 'V', 'Includes 2x multiplier'])
            writer.writerow(['Voltage Range', voltage_range, 'V', 'Max - Min'])
            writer.writerow(['Voltage Standard Deviation', voltage_std_dev, 'V', 'Signal variation'])

        print(f"Analysis data saved to: {CSV_FILE_ANALYSIS}")
        
        # Save analysis to text file
        analysis_data = {
            'pulse_intervals': pulse_intervals,
            'avg_period': avg_period,
            'std_dev': std_dev,
            'min_period': min_period,
            'max_period': max_period,
            'pulse_count': pulse_count,
            'mean_voltage': mean_voltage,
            'min_voltage': min_voltage,
            'max_voltage': max_voltage,
            'voltage_range': voltage_range,
            'voltage_std_dev': voltage_std_dev
        }
        txt_file = save_analysis_to_txt(analysis_data, file_prefix, session_path)
        
        # Generate statistical plots
        if len(timestamps) > 0:
            plot_file = plot_statistics(
                analysis_data,
                timestamps, 
                voltages,  # Use voltage instead of ADC values
                pulses, 
                delta_times,
                file_prefix,
                session_path
            )
            
            # Ask if user wants to view the plots now
            view_plots = input("\nWould you like to view the plots now? (y/n): ")
            if view_plots.lower() == 'y':
                plt.show()
            
    else:
        print("No pulses were detected in the session.")
        
    print(f"\nSession data saved to folder: {session_path}")

if __name__ == "__main__":
    main()
