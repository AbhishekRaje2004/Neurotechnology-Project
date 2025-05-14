import serial
import csv
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from serial.tools import list_ports

# === CONFIGURATION ===
BAUD_RATE = 500000
VREF = 3.3
MAX_ADC = 4095
THRESHOLD_V = 0.6
MIN_PULSE_US = 50
MAX_PULSE_US = 1000
EXPECTED_WIDTH_US = 400.0  # Updated to expected value of 400µs
TRAILING_SAMPLES = 5
SAMPLE_RATE_ESTIMATION_COUNT = 100
DEFAULT_DURATION_SECONDS = 60  # Default duration if user doesn't specify
VOLTAGE_MULTIPLIER = 2.0  # True voltage is twice the measured value

# === USER INTERFACE FUNCTIONS ===
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
    default_prefix = "waveform"
    prefix = input(f"\nEnter filename prefix (default: '{default_prefix}'): ")
    if not prefix:
        return default_prefix
    return prefix

def get_session_folder_name():
    """Get a name for the session folder"""
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    default_name = f"waveform_session_{timestamp_str}"
    
    print("\nSession Folder Name")
    print("------------------")
    folder_name = input(f"Enter a name for this recording session (default: '{default_name}'): ")
    
    if not folder_name:
        return default_name
    
    # Remove invalid characters for folders
    folder_name = "".join([c for c in folder_name if c.isalnum() or c in " _-"])
    return folder_name

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

# === ANALYSIS FUNCTIONS ===
def estimate_sampling_interval_us(timestamps):
    deltas = np.diff(timestamps)
    avg_us = np.mean(deltas)
    print(f"Estimated sampling rate: {1e6 / avg_us:.2f} Hz ({avg_us:.1f} µs/sample)")
    return avg_us

def analyze_adc_data(df, session_path, file_prefix):
    """Analyze waveform data and generate reports and visualizations"""
    t = df['timestamp_us'].to_numpy()
    v = df['voltage_V'].to_numpy()
    
    # Apply voltage multiplier to get true voltage values
    v = v * VOLTAGE_MULTIPLIER

    print(f"\nAnalyzing {len(t)} samples...")
    avg_interval_us = estimate_sampling_interval_us(t[:min(len(t), SAMPLE_RATE_ESTIMATION_COUNT)])

    in_pulse = False
    pulse_data = []
    pulse_stats = []
    
    # Create plots directory
    plots_dir = os.path.join(session_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("\nDetecting pulses...")
    for i in range(len(v)):
        if not in_pulse and v[i] >= THRESHOLD_V * VOLTAGE_MULTIPLIER:
            # Pulse starts at this sample
            pulse_start_idx = i
            in_pulse = True
            pulse_data = [(t[i], v[i])]
        elif in_pulse:
            pulse_data.append((t[i], v[i]))

            if v[i] < THRESHOLD_V * VOLTAGE_MULTIPLIER:
                # Pulse ends at this sample
                pulse_end_idx = i
                if len(pulse_data) >= TRAILING_SAMPLES:
                    trailing = [v_ for _, v_ in pulse_data[-TRAILING_SAMPLES:]]
                    if all(v_ < THRESHOLD_V * VOLTAGE_MULTIPLIER for v_ in trailing):
                        timestamps, volts = zip(*pulse_data)
                        timestamps = np.array(timestamps)
                        volts = np.array(volts)
                        
                        # Safe duration calculation using timestamps directly
                        duration = timestamps[-1] - timestamps[0]
                        
                        # Only process pulses within expected range
                        if MIN_PULSE_US <= duration <= MAX_PULSE_US:
                            # Find peak within this pulse's data only
                            local_peak_idx = np.argmax(volts)
                            peak = volts[local_peak_idx]
                            
                            # Calculate rise and fall times using local indices
                            rise_time = timestamps[local_peak_idx] - timestamps[0]
                            fall_time = timestamps[-1] - timestamps[local_peak_idx]
                            
                            error = 100 * (duration - EXPECTED_WIDTH_US) / EXPECTED_WIDTH_US

                            pulse_stats.append({
                                "duration": duration,
                                "rise_time": rise_time,
                                "fall_time": fall_time,
                                "peak": peak,
                                "local_peak_idx": local_peak_idx,  # Store local index (within pulse data)
                                "error": error,
                                "data": (timestamps, volts),
                                "num_samples": len(timestamps)
                            })

                            # Print pulse info
                            pulse_num = len(pulse_stats)
                            print(f"Pulse #{pulse_num}: Width={duration:.1f}µs, Peak={peak:.2f}V, Rise={rise_time:.1f}µs, Fall={fall_time:.1f}µs, Samples={len(timestamps)}")

                        in_pulse = False
                        pulse_data = []

    # Create pulse analysis plots if pulses detected
    if pulse_stats:
        # Generate a summary plot of all detected pulses
        plt.figure(figsize=(10, 6))
        
        for i, pulse in enumerate(pulse_stats):
            timestamps, volts = pulse["data"]
            rel_t = timestamps - timestamps[0]
            plt.plot(rel_t, volts, label=f"Pulse {i+1}" if i < 10 else None)
            
            # Mark the peak point using local peak index
            local_peak_idx = pulse["local_peak_idx"]
            plt.plot(rel_t[local_peak_idx], volts[local_peak_idx], 'ro', markersize=4)
        
        plt.axhline(THRESHOLD_V * VOLTAGE_MULTIPLIER, color='red', linestyle='--', label="Threshold")
        plt.title("Detected Pulses", fontsize=14)
        plt.xlabel("Time (µs)", fontsize=12)
        plt.ylabel("Voltage (V)", fontsize=12)
        plt.grid(True, alpha=0.3)
        if len(pulse_stats) <= 10:
            plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save the plot
        all_pulses_file = os.path.join(plots_dir, f"{file_prefix}_all_pulses.png")
        plt.savefig(all_pulses_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Generate distribution plots
        widths = [p["duration"] for p in pulse_stats]
        rise_times = [p["rise_time"] for p in pulse_stats]
        fall_times = [p["fall_time"] for p in pulse_stats]
        peaks = [p["peak"] for p in pulse_stats]
        sample_counts = [p["num_samples"] for p in pulse_stats]
        
        # Width histogram
        plt.figure(figsize=(10, 6))
        plt.hist(widths, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
        plt.axvline(x=EXPECTED_WIDTH_US, color='r', linestyle='--', 
                   label=f"Expected ({EXPECTED_WIDTH_US} µs)")
        plt.axvline(x=np.mean(widths), color='g', linestyle='-',
                   label=f"Mean ({np.mean(widths):.1f} µs)")
        plt.title("Pulse Width Distribution", fontsize=14)
        plt.xlabel("Pulse Width (µs)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Save the plot
        width_hist_file = os.path.join(plots_dir, f"{file_prefix}_width_hist.png")
        plt.savefig(width_hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Rise/Fall Time plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Rise time histogram
        ax1.hist(rise_times, bins=15, color='green', edgecolor='black', alpha=0.7)
        ax1.axvline(x=np.mean(rise_times), color='r', linestyle='-',
                   label=f"Mean ({np.mean(rise_times):.1f} µs)")
        ax1.set_title("Rise Time Distribution (First Crossing to Peak)", fontsize=14)
        ax1.set_xlabel("Rise Time (µs)", fontsize=12)
        ax1.set_ylabel("Frequency", fontsize=12)
        ax1.legend(fontsize=10)
        ax1.grid(True, axis='y', alpha=0.3)
        
        # Fall time histogram
        ax2.hist(fall_times, bins=15, color='orange', edgecolor='black', alpha=0.7)
        ax2.axvline(x=np.mean(fall_times), color='r', linestyle='-',
                   label=f"Mean ({np.mean(fall_times):.1f} µs)")
        ax2.set_title("Fall Time Distribution (Peak to Last Crossing)", fontsize=14)
        ax2.set_xlabel("Fall Time (µs)", fontsize=12)
        ax2.set_ylabel("Frequency", fontsize=12)
        ax2.legend(fontsize=10)
        ax2.grid(True, axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        timing_hist_file = os.path.join(plots_dir, f"{file_prefix}_timing_hist.png")
        plt.savefig(timing_hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Peak voltage distribution
        plt.figure(figsize=(10, 6))
        plt.hist(peaks, bins=20, color='purple', edgecolor='black', alpha=0.7)
        plt.axvline(x=np.mean(peaks), color='r', linestyle='-',
                   label=f"Mean ({np.mean(peaks):.2f} V)")
        plt.title("Peak Voltage Distribution", fontsize=14)
        plt.xlabel("Voltage (V)", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        
        # Save the plot
        voltage_hist_file = os.path.join(plots_dir, f"{file_prefix}_voltage_hist.png")
        plt.savefig(voltage_hist_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Samples per pulse distribution
        plt.figure(figsize=(10, 6))
        plt.hist(sample_counts, bins=range(min(sample_counts), max(sample_counts) + 2), color='teal', edgecolor='black', alpha=0.7)
        plt.axvline(x=np.mean(sample_counts), color='r', linestyle='-',
                   label=f"Mean ({np.mean(sample_counts):.1f} samples)")
        plt.title("Samples Per Pulse Distribution", fontsize=14)
        plt.xlabel("Number of Samples", fontsize=12)
        plt.ylabel("Frequency", fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, axis='y', alpha=0.3)
        plt.xticks(range(min(sample_counts), max(sample_counts) + 1))  # Integer x-axis
        
        # Save the plot
        samples_hist_file = os.path.join(plots_dir, f"{file_prefix}_samples_hist.png")
        plt.savefig(samples_hist_file, dpi=300, bbox_inches='tight')
        plt.close()
    
    # Save results as CSV
    csv_file = os.path.join(session_path, f"{file_prefix}_analysis.csv")
    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Pulse#", "Width(µs)", "Rise Time(µs)", "Fall Time(µs)", "Peak(V)", "Error(%)", "Samples"])
        for i, pulse in enumerate(pulse_stats):
            writer.writerow([
                i+1, 
                pulse["duration"], 
                pulse["rise_time"],
                pulse["fall_time"],
                pulse["peak"],
                pulse["error"],
                pulse["num_samples"]
            ])
    
    # Generate text report
    txt_report = os.path.join(session_path, f"{file_prefix}_analysis.txt")
    with open(txt_report, 'w') as f:
        f.write("WAVEFORM DATA ANALYSIS REPORT\n")
        f.write("============================\n\n")
        f.write(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data collection session: {os.path.basename(session_path)}\n")
        f.write(f"Total samples analyzed: {len(t)}\n")
        f.write(f"Note: Voltage values reflect actual voltages (2x the measured values)\n\n")
        
        if pulse_stats:
            # Calculate statistics
            widths = [p["duration"] for p in pulse_stats]
            rise_times = [p["rise_time"] for p in pulse_stats]
            fall_times = [p["fall_time"] for p in pulse_stats]
            peaks = [p["peak"] for p in pulse_stats]
            sample_counts = [p["num_samples"] for p in pulse_stats]
            
            avg_width = np.mean(widths)
            std_width = np.std(widths)
            avg_error = np.mean([p["error"] for p in pulse_stats])
                        
            # Write pulse count and general info
            f.write(f"\nTotal pulses detected: {len(pulse_stats)}\n")
            f.write(f"Sample interval: {avg_interval_us:.2f} µs\n")
            f.write(f"Expected pulse width: {EXPECTED_WIDTH_US:.1f} µs\n")
            f.write(f"Average samples per pulse: {np.mean(sample_counts):.1f}\n\n")
            
            # Write width statistics
            f.write("PULSE WIDTH STATISTICS\n")
            f.write("---------------------\n")
            f.write(f"Average width: {avg_width:.2f} µs\n")
            f.write(f"Width std dev: {std_width:.2f} µs\n")
            f.write(f"Min width: {min(widths):.2f} µs\n")
            f.write(f"Max width: {max(widths):.2f} µs\n")
            f.write(f"Avg error from expected: {avg_error:+.2f}%\n\n")
            
            # Write timing statistics 
            f.write("TIMING STATISTICS\n")
            f.write("----------------\n")
            f.write("Note: Rise time measured from first crossing to peak\n")
            f.write("      Fall time measured from peak to first exit below threshold\n\n")
            f.write(f"Average rise time: {np.mean(rise_times):.2f} µs\n")
            f.write(f"Rise time std dev: {np.std(rise_times):.2f} µs\n")
            f.write(f"Average fall time: {np.mean(fall_times):.2f} µs\n")
            f.write(f"Fall time std dev: {np.std(fall_times):.2f} µs\n\n")
            
            # Write voltage statistics
            f.write("VOLTAGE STATISTICS\n")
            f.write("-----------------\n")
            f.write(f"Average peak voltage: {np.mean(peaks):.3f} V\n")
            f.write(f"Peak voltage std dev: {np.std(peaks):.3f} V\n")
            f.write(f"Min peak voltage: {min(peaks):.3f} V\n")
            f.write(f"Max peak voltage: {max(peaks):.3f} V\n\n")
            
            # Write first 10 pulse details
            f.write("INDIVIDUAL PULSE DETAILS (First 10)\n")
            f.write("---------------------------------\n")
            for i, pulse in enumerate(pulse_stats[:10]):
                f.write(f"Pulse #{i+1}:\n")
                f.write(f"  Width: {pulse['duration']:.2f} µs\n")
                f.write(f"  Rise time: {pulse['rise_time']:.2f} µs (to peak)\n")
                f.write(f"  Fall time: {pulse['fall_time']:.2f} µs (from peak)\n")
                f.write(f"  Peak voltage: {pulse['peak']:.3f} V\n")
                f.write(f"  Samples in pulse: {pulse['num_samples']}\n") 
                f.write(f"  Error vs expected: {pulse['error']:+.2f}%\n\n")
        else:
            f.write("\nNo valid pulses detected in the data.\n")
            f.write("Try adjusting the threshold or pulse width parameters.\n")
    
    # Create a plots index file
    if pulse_stats:
        plots_index = os.path.join(session_path, f"{file_prefix}_plots_index.txt")
        with open(plots_index, 'w') as f:
            f.write("WAVEFORM DATA VISUALIZATION FILES\n")
            f.write("===============================\n\n")
            f.write(f"Session: {os.path.basename(session_path)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"1. All Pulses Plot\n")
            f.write(f"   Path: plots/{os.path.basename(all_pulses_file)}\n\n")
            
            f.write(f"2. Pulse Width Distribution\n")
            f.write(f"   Path: plots/{os.path.basename(width_hist_file)}\n\n")
            
            f.write(f"3. Rise/Fall Time Distribution\n")
            f.write(f"   Path: plots/{os.path.basename(timing_hist_file)}\n\n")
            
            f.write(f"4. Peak Voltage Distribution\n")
            f.write(f"   Path: plots/{os.path.basename(voltage_hist_file)}\n\n")
            
            f.write(f"5. Samples Per Pulse Distribution\n")
            f.write(f"   Path: plots/{os.path.basename(samples_hist_file)}\n\n")
    
    return pulse_stats

# === MAIN FUNCTION ===
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
    csv_file = os.path.join(session_path, f"{file_prefix}_data_raw.csv")
    csv_corrupted = os.path.join(session_path, f"{file_prefix}_data_corrupted.csv")
    
    print(f"\nConnecting to {port} @ {BAUD_RATE} baud...")
    
    try:
        with serial.Serial(port, BAUD_RATE, timeout=1) as ser, \
             open(csv_file, 'w', newline='') as f, \
             open(csv_corrupted, 'w', newline='') as f_corr:
             
            writer = csv.writer(f)
            writer_corrupted = csv.writer(f_corr)
            writer.writerow(["timestamp_us", "adc_value", "voltage_V"])
            writer_corrupted.writerow(["RawLine"])
            
            print(f"\nReading waveform data from {port} for {duration_seconds} seconds...")
            print("Press Ctrl+C to stop early.")
            
            # Setup timing
            start_time = time.time()
            end_time = start_time + duration_seconds
            last_progress_update = start_time
            sample_count = 0

            while time.time() < end_time:
                # Show progress every second
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = duration_seconds - elapsed
                
                if current_time - last_progress_update >= 1:
                    progress_pct = int((elapsed / duration_seconds) * 100)
                    print(f"\rProgress: {progress_pct}% - {int(remaining)}s remaining - {sample_count} samples", end='')
                    last_progress_update = current_time
                
                # Check if data is available to read
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8', errors='ignore').strip()
                    
                    if line.startswith("ADC"):
                        try:
                            _, t_str, v_str = line.split(",")
                            t_us = int(t_str)
                            adc_val = int(v_str)
                            voltage = adc_val * VREF / MAX_ADC
                            writer.writerow([t_us, adc_val, voltage])
                            sample_count += 1
                        except (ValueError, IndexError):
                            writer_corrupted.writerow([line])
                    else:
                        writer_corrupted.writerow([line])
                else:
                    # Sleep briefly to prevent CPU hogging if no data
                    time.sleep(0.01)
            
            print("\nData collection complete!")
            
    except KeyboardInterrupt:
        elapsed = time.time() - start_time
        print(f"\nData collection stopped by user after {int(elapsed)} seconds.")
        
    except serial.SerialException as e:
        print(f"Error with serial connection: {e}")
        return
    
    print(f"\nData collection finished. {sample_count} samples saved to {csv_file}")
    
    # Ask if user wants to analyze now
    analyze_now = input("\nAnalyze the collected data now? (y/n): ")
    if analyze_now.lower() == 'y':
        print("\nLoading data for analysis...")
        try:
            df = pd.read_csv(csv_file)
            if len(df) > 0:
                pulse_stats = analyze_adc_data(df, session_path, file_prefix)
                
                if pulse_stats:
                    print(f"\nAnalysis complete! {len(pulse_stats)} pulses detected and analyzed.")
                    print(f"Reports and visualizations saved to: {session_path}")
                else:
                    print("\nNo valid pulses detected in the data.")
            else:
                print("\nNo data found in the CSV file.")
        except Exception as e:
            print(f"\nError during analysis: {e}")
    
    print(f"\nSession data saved to folder: {session_path}")

if __name__ == "__main__":
    main()
