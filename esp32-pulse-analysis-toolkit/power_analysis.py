import serial
import csv
import os
import time
import statistics
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from serial.tools import list_ports

# === Configuration ===
BAUD_RATE = 115200
DEFAULT_DURATION_SECONDS = 60  # Default duration if user doesn't specify
DEFAULT_VOLTAGE_MULTIPLIER = 2.0  # Default voltage multiplier

# ESP32 ADC Configuration
ADC_RESOLUTION = 12  # 12-bit ADC
ADC_MAX_VALUE = (2**ADC_RESOLUTION) - 1  # 4095 for 12-bit
ADC_VOLTAGE_REFERENCE = 3.3  # ESP32 voltage reference is 3.3V

def adc_to_voltage(adc_value):
    """Convert ADC reading to voltage value, applying the multiplier"""
    raw_voltage = (adc_value / ADC_MAX_VALUE) * ADC_VOLTAGE_REFERENCE
    return raw_voltage * DEFAULT_VOLTAGE_MULTIPLIER

# === Line parser ===
def parse_line(line):
    """Parse a line from the serial port, handling different formats"""
    try:
        parts = line.strip().split(',')
        
        # Handle format from waveform_capture.ino: "ADC,timestamp,adc_value"
        if len(parts) == 3 and parts[0] == "ADC":
            timestamp = int(parts[1])
            adc_value = int(parts[2])
            voltage = adc_to_voltage(adc_value)
            
            # For this format, we don't have pulse detection or delta
            # So we'll create simulated ones based on voltage threshold
            pulse_detected = 1 if adc_value > 500 else 0  # Arbitrary threshold
            delta = 0  # No timing information available
            
            return {
                'timestamp': timestamp,
                'adc_value': adc_value,
                'voltage': voltage,
                'pulse': pulse_detected,
                'delta': delta,
                'format': 'waveform'  # Tag to identify the format
            }
        
        # Handle format from pulse_detector.ino: "timestamp,adc_value,pulse_detected,delta"
        elif len(parts) == 4:
            adc_value = int(parts[1])
            return {
                'timestamp': int(parts[0]),
                'adc_value': adc_value,
                'voltage': adc_to_voltage(adc_value),
                'pulse': int(parts[2]),
                'delta': int(parts[3]),
                'format': 'pulse'  # Tag to identify the format
            }
        
        return None
    except ValueError:
        return None

# === User Interface Functions ===
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
    default_prefix = "power"
    prefix = input(f"\nEnter filename prefix (default: '{default_prefix}'): ")
    if not prefix:
        return default_prefix
    return prefix

def get_session_folder_name():
    """Get a name for the session folder"""
    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    # Update default name to include power_ prefix
    default_name = f"power_session_{timestamp_str}"
    
    print("\nSession Folder Name")
    print("------------------")
    folder_name = input(f"Enter a name for this recording session (default: '{default_name}'): ")
    
    if not folder_name:
        return default_name
    
    # Remove invalid characters for folders
    folder_name = "".join([c for c in folder_name if c.isalnum() or c in " _-"])
    
    # If user provided a custom name without prefix, add the power_ prefix
    if not folder_name.startswith(('power_', 'pulse_', 'waveform_')):
        folder_name = f"power_{folder_name}"
        
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

def get_pulse_voltage():
    """Get the pulse voltage from the user"""
    while True:
        try:
            voltage = float(input("\nEnter Pulse Voltage (V): "))
            if voltage <= 0:
                print("Voltage must be positive. Please try again.")
                continue
            return voltage
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_resistor_value():
    """Get the known resistor value from the user"""
    default_value = 470.0
    while True:
        try:
            value_input = input(f"\nEnter Known Resistor Value (ohm, default: {default_value}): ")
            if not value_input:
                return default_value
            
            value = float(value_input)
            if value <= 0:
                print("Resistance must be positive. Please try again.")
                continue
            return value
        except ValueError:
            print("Invalid input. Please enter a number.")

def generate_plots(data, session_path, file_prefix):
    """Generate plots for power analysis data"""
    if not data or not data['timestamps']:
        print("No data to plot")
        return None
        
    # Create plots directory
    plots_dir = os.path.join(session_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    plot_files = []
    
    # === Plot 1: Resistance over time ===
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestamps'], data['resistances'], 'b-', linewidth=2)
    plt.title('Internal Resistance Over Time', fontsize=14)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Internal Resistance (ohm)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    resistance_plot_file = os.path.join(plots_dir, f'{file_prefix}_resistance.png')
    plt.savefig(resistance_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(resistance_plot_file)
    
    # === Plot 2: Power over time ===
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestamps'], data['powers'], 'r-', linewidth=2)
    plt.title('Power Consumption Over Time', fontsize=14) 
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Power (W)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    power_plot_file = os.path.join(plots_dir, f'{file_prefix}_power.png')
    plt.savefig(power_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(power_plot_file)
    
    # === Plot 3: Current over time ===
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestamps'], data['currents'], 'g-', linewidth=2)
    plt.title('Current Over Time', fontsize=14)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Current (A)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    current_plot_file = os.path.join(plots_dir, f'{file_prefix}_current.png')
    plt.savefig(current_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(current_plot_file)

    # === Plot 4: Voltage over time (new) ===
    plt.figure(figsize=(10, 6))
    plt.plot(data['timestamps'], data['voltages'], color='purple', linewidth=2)
    plt.title('Measured Voltage Over Time', fontsize=14)
    plt.xlabel('Time (ms)', fontsize=12)
    plt.ylabel('Voltage (V)', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    voltage_plot_file = os.path.join(plots_dir, f'{file_prefix}_voltage.png')
    plt.savefig(voltage_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(voltage_plot_file)
    
    # === Individual Distribution Histograms ===
    
    # Resistance histogram (separate file)
    plt.figure(figsize=(10, 6))
    plt.hist(data['resistances'], bins=20, color='blue', alpha=0.7)
    plt.axvline(x=statistics.mean(data['resistances']), color='r', linestyle='--', 
               label=f'Mean: {statistics.mean(data["resistances"]):.2f} ohm')
    plt.title('Internal Resistance Distribution', fontsize=14)
    plt.xlabel('Resistance (ohm)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the resistance histogram
    resistance_hist_file = os.path.join(plots_dir, f'{file_prefix}_resistance_hist.png')
    plt.savefig(resistance_hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(resistance_hist_file)
    
    # Power histogram (separate file)
    plt.figure(figsize=(10, 6))
    plt.hist(data['powers'], bins=20, color='red', alpha=0.7)
    plt.axvline(x=statistics.mean(data['powers']), color='r', linestyle='--',
               label=f'Mean: {statistics.mean(data["powers"]):.6f} W')
    plt.title('Power Distribution', fontsize=14)
    plt.xlabel('Power (W)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the power histogram
    power_hist_file = os.path.join(plots_dir, f'{file_prefix}_power_hist.png')
    plt.savefig(power_hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(power_hist_file)
    
    # Current histogram (separate file)
    plt.figure(figsize=(10, 6))
    plt.hist(data['currents'], bins=20, color='green', alpha=0.7)
    plt.axvline(x=statistics.mean(data['currents']), color='r', linestyle='--',
               label=f'Mean: {statistics.mean(data["currents"]):.6f} A')
    plt.title('Current Distribution', fontsize=14)
    plt.xlabel('Current (A)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the current histogram
    current_hist_file = os.path.join(plots_dir, f'{file_prefix}_current_hist.png')
    plt.savefig(current_hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(current_hist_file)
    
    # Voltage histogram (separate file)
    plt.figure(figsize=(10, 6))
    plt.hist(data['voltages'], bins=20, color='purple', alpha=0.7)
    plt.axvline(x=statistics.mean(data['voltages']), color='r', linestyle='--',
               label=f'Mean: {statistics.mean(data["voltages"]):.3f} V')
    plt.title('Measured Voltage Distribution', fontsize=14)
    plt.xlabel('Voltage (V)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Save the voltage histogram
    voltage_hist_file = os.path.join(plots_dir, f'{file_prefix}_voltage_hist.png')
    plt.savefig(voltage_hist_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(voltage_hist_file)
    
    # === Combined Distribution Histograms (keep original) ===
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    
    # Resistance histogram
    ax1.hist(data['resistances'], bins=20, color='blue', alpha=0.7)
    ax1.axvline(x=statistics.mean(data['resistances']), color='r', linestyle='--', 
               label=f'Mean: {statistics.mean(data["resistances"]):.2f} ohm')
    ax1.set_title('Internal Resistance Distribution')
    ax1.set_xlabel('Resistance (ohm)')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    
    # Power histogram
    ax2.hist(data['powers'], bins=20, color='red', alpha=0.7)
    ax2.axvline(x=statistics.mean(data['powers']), color='r', linestyle='--',
               label=f'Mean: {statistics.mean(data["powers"]):.6f} W')
    ax2.set_title('Power Distribution')
    ax2.set_xlabel('Power (W)')
    ax2.set_ylabel('Frequency')
    ax2.legend()
    
    # Current histogram
    ax3.hist(data['currents'], bins=20, color='green', alpha=0.7)
    ax3.axvline(x=statistics.mean(data['currents']), color='r', linestyle='--',
               label=f'Mean: {statistics.mean(data["currents"]):.6f} A')
    ax3.set_title('Current Distribution')
    ax3.set_xlabel('Current (A)')
    ax3.set_ylabel('Frequency')
    ax3.legend()
    
    plt.tight_layout()
    
    # Save the plot
    dist_plot_file = os.path.join(plots_dir, f'{file_prefix}_distributions.png')
    plt.savefig(dist_plot_file, dpi=300, bbox_inches='tight')
    plt.close()
    plot_files.append(dist_plot_file)
    
    # Create a plots index file
    plots_index = os.path.join(session_path, f'{file_prefix}_plots_index.txt')
    with open(plots_index, 'w') as f:
        f.write("POWER ANALYSIS VISUALIZATION FILES\n")
        f.write("===============================\n\n")
        f.write(f"Session: {os.path.basename(session_path)}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("TIME SERIES PLOTS\n")
        f.write("--------------\n")
        f.write(f"1. Internal Resistance Plot\n")
        f.write(f"   Path: plots/{os.path.basename(resistance_plot_file)}\n\n")
        
        f.write(f"2. Power Consumption Plot\n")
        f.write(f"   Path: plots/{os.path.basename(power_plot_file)}\n\n")
        
        f.write(f"3. Current Plot\n")
        f.write(f"   Path: plots/{os.path.basename(current_plot_file)}\n\n")
        
        f.write(f"4. Voltage Plot\n")
        f.write(f"   Path: plots/{os.path.basename(voltage_plot_file)}\n\n")
        
        f.write("DISTRIBUTION HISTOGRAMS\n")
        f.write("----------------------\n")
        f.write(f"5. Internal Resistance Histogram\n")
        f.write(f"   Path: plots/{os.path.basename(resistance_hist_file)}\n\n")
        
        f.write(f"6. Power Distribution Histogram\n")
        f.write(f"   Path: plots/{os.path.basename(power_hist_file)}\n\n")
        
        f.write(f"7. Current Distribution Histogram\n")
        f.write(f"   Path: plots/{os.path.basename(current_hist_file)}\n\n")
        
        f.write(f"8. Voltage Distribution Histogram\n")
        f.write(f"   Path: plots/{os.path.basename(voltage_hist_file)}\n\n")
        
        f.write(f"9. Combined Distribution Histograms\n")
        f.write(f"   Path: plots/{os.path.basename(dist_plot_file)}\n\n")
    
    return plot_files

def save_analysis_report(data, session_path, file_prefix, config):
    """Save analysis results to a human-readable text file"""
    txt_file = os.path.join(session_path, f'{file_prefix}_analysis.txt')
    
    with open(txt_file, 'w', encoding='utf-8') as f:
        f.write("POWER ANALYSIS REPORT\n")
        f.write("====================\n\n")
        f.write(f"Analysis performed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Data collection session: {os.path.basename(session_path)}\n")
        f.write(f"Total pulses detected: {len(data['timestamps'])}\n\n")
        
        f.write("CONFIGURATION\n")
        f.write("-------------\n")
        f.write(f"Pulse Voltage: {config['pulse_voltage']:.3f} V\n")
        f.write(f"Known Resistor: {config['r_known']:.1f} ohm\n")
        f.write(f"Total Known Resistance: {config['r_total_known']:.1f} ohm\n\n")
        
        if data['timestamps']:
            f.write("INTERNAL RESISTANCE STATISTICS\n")
            f.write("-----------------------------\n")
            f.write(f"Average: {statistics.mean(data['resistances']):.2f} ohm\n")
            f.write(f"Minimum: {min(data['resistances']):.2f} ohm\n")
            f.write(f"Maximum: {max(data['resistances']):.2f} ohm\n")
            f.write(f"Standard Deviation: {statistics.stdev(data['resistances']):.2f} ohm\n\n")
            
            f.write("POWER STATISTICS\n")
            f.write("---------------\n")
            f.write(f"Average Power: {statistics.mean(data['powers']):.6f} W\n")
            f.write(f"Minimum Power: {min(data['powers']):.6f} W\n")
            f.write(f"Maximum Power: {max(data['powers']):.6f} W\n")
            f.write(f"Standard Deviation: {statistics.stdev(data['powers']):.6f} W\n\n")
            
            f.write("CURRENT STATISTICS\n")
            f.write("-----------------\n")
            f.write(f"Average Current: {statistics.mean(data['currents']):.6f} A\n")
            f.write(f"Minimum Current: {min(data['currents']):.6f} A\n")
            f.write(f"Maximum Current: {max(data['currents']):.6f} A\n")
            f.write(f"Standard Deviation: {statistics.stdev(data['currents']):.6f} A\n\n")
            
            # Energy calculation (power * time)
            total_energy = 0
            for i in range(1, len(data['timestamps'])):
                dt = (data['timestamps'][i] - data['timestamps'][i-1]) / 1000  # Convert to seconds
                avg_power = (data['powers'][i] + data['powers'][i-1]) / 2
                total_energy += avg_power * dt
                
            f.write("ENERGY CONSUMPTION\n")
            f.write("-----------------\n")
            f.write(f"Total Energy: {total_energy:.6f} J\n")
            f.write(f"Session Duration: {(data['timestamps'][-1] - data['timestamps'][0])/1000:.2f} s\n")
                
    print(f"Analysis report saved to: {txt_file}")
    return txt_file

# === Main function ===
def main():
    print("=== ESP32 Pulse Power Analyzer ===")
    
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
    
    # Get the pulse voltage
    pulse_voltage = get_pulse_voltage()
    
    # Get the known resistor value
    r_known = get_resistor_value()
    r_total_known = 2 * r_known  # two resistors in series
    
    # Get desired duration
    duration_seconds = get_duration()
    print(f"\nData will be collected for {duration_seconds} seconds.")
    
    # Define output files
    csv_file = os.path.join(session_path, f"{file_prefix}_data_raw.csv")
    csv_corrupted = os.path.join(session_path, f"{file_prefix}_data_corrupted.csv")
    
    # Storage for power analysis data
    data = {
        'timestamps': [],
        'voltages': [],
        'resistances': [],
        'currents': [],
        'powers': []
    }
    
    print(f"\nConnecting to {port} @ {BAUD_RATE} baud...")
    
    try:
        with serial.Serial(port, BAUD_RATE, timeout=1) as ser, \
             open(csv_file, 'w', newline='') as f, \
             open(csv_corrupted, 'w', newline='') as f_corr:
             
            writer = csv.writer(f)
            writer_corrupted = csv.writer(f_corr)
            writer.writerow([
                'Timestamp(ms)', 
                'ADC_Value', 
                'Voltage(V)', 
                'PulseDetected', 
                'InternalResistance(ohm)', 
                'Current(A)', 
                'Power(W)'
            ])
            writer_corrupted.writerow(['RawLine'])
            
            # Flush input buffer
            ser.flushInput()
            
            print(f"\nReading pulse data from {port} for {duration_seconds} seconds...")
            print("Press Ctrl+C to stop early.")
            
            # Setup timing
            start_time = time.time()
            end_time = start_time + duration_seconds
            last_progress_update = start_time
            pulse_count = 0
            
            while time.time() < end_time:
                # Show progress every second
                current_time = time.time()
                elapsed = current_time - start_time
                remaining = duration_seconds - elapsed
                
                if current_time - last_progress_update >= 1:
                    progress_pct = int((elapsed / duration_seconds) * 100)
                    print(f"\rProgress: {progress_pct}% - {int(remaining)}s remaining - {pulse_count} pulses detected", end='')
                    last_progress_update = current_time
                
                # Check if data is available to read
                if ser.in_waiting > 0:
                    try:
                        # Read and decode line safely
                        line = ser.readline().decode('utf-8', errors='ignore').strip()
                        
                        # Parse the serial data
                        parsed_data = parse_line(line)
                        if parsed_data is None:
                            writer_corrupted.writerow([line])
                            continue
                        
                        timestamp = parsed_data['timestamp']
                        adc_val = parsed_data['adc_value']
                        v_measured = parsed_data['voltage']
                        pulse_detected = parsed_data['pulse']
                        delta = parsed_data['delta']
                        
                        # Check if this is a valid pulse
                        if adc_val > 0:  # Just check if we have any reading at all
                            try:
                                # Estimate internal resistance
                                if v_measured > 0.01:  # Lower threshold to catch more data
                                    r_internal = (r_known * (pulse_voltage / v_measured)) - r_total_known
                                    
                                    # Accept negative resistances for diagnostic purposes
                                    # (will indicate issues with voltage settings)
                                    if r_internal <= 0:
                                        # Use absolute value for calculations
                                        r_internal = abs(r_internal)
                                    
                                    # Calculate current and power
                                    total_resistance = r_total_known + r_internal
                                    current = pulse_voltage / total_resistance
                                    # Corrected power calculation: P = V^2/R
                                    power = (pulse_voltage * pulse_voltage) / r_internal
                                    
                                    # Store data for analysis
                                    data['timestamps'].append(timestamp)
                                    data['voltages'].append(v_measured)
                                    data['resistances'].append(r_internal)
                                    data['currents'].append(current)
                                    data['powers'].append(power)
                                    
                                    # Write to CSV
                                    writer.writerow([
                                        timestamp, 
                                        adc_val, 
                                        v_measured, 
                                        pulse_detected, 
                                        r_internal, 
                                        current, 
                                        power
                                    ])
                                    
                                    pulse_count += 1
                            except Exception as calc_error:
                                # Silently log to corrupted file instead of printing
                                writer_corrupted.writerow([f"Calculation error: {calc_error}"])
                    except Exception as e:
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
    
    print(f"\nData collection finished. {pulse_count} pulses saved to {csv_file}")
    
    # Create analysis report and plots if we have data
    if data['timestamps']:
        config = {
            'pulse_voltage': pulse_voltage,
            'r_known': r_known,
            'r_total_known': r_total_known
        }
        
        # Generate analysis report
        save_analysis_report(data, session_path, file_prefix, config)
        
        # Generate plots
        plot_files = generate_plots(data, session_path, file_prefix)
        
        print(f"\nAnalysis complete! {len(data['timestamps'])} pulses analyzed.")
        print(f"Reports and visualizations saved to: {session_path}")
        
        # Ask if user wants to view the plots now
        view_plots = input("\nWould you like to view the plots now? (y/n): ")
        if view_plots.lower() == 'y':
            plt.figure(figsize=(10, 6))
            plt.plot(data['timestamps'], data['powers'], 'r-', linewidth=2)
            plt.title('Power Consumption Over Time', fontsize=14) 
            plt.xlabel('Time (ms)', fontsize=12)
            plt.ylabel('Power (W)', fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.show()
    else:
        print("\nNo pulses were detected in the session.")
    
    print(f"\nSession data saved to folder: {session_path}")


if __name__ == "__main__":
    main()
