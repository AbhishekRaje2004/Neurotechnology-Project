# ESP32 Pulse Analysis Toolkit

A comprehensive toolkit for analyzing electrical pulses, power consumption, and waveforms using an ESP32 microcontroller.

## Overview

This toolkit consists of three main components:

1. **Pulse Detection** - Detects and analyzes pulse timing and characteristics
2. **Power Analysis** - Measures power consumption, internal resistance, and current
3. **Waveform Capture** - Captures detailed waveform data for analysis

## Hardware Requirements

- ESP32 development board
- Appropriate voltage divider resistors (470 ohm recommended)
- External power source for testing
- USB cable for connecting ESP32 to computer

## Software Requirements

Install the required Python packages:

```
pip install -r requirements.txt
```

## Setup Instructions

### 1. Upload Arduino Sketches

Two Arduino sketches are provided:

- **pulse_detector.ino** - Used for both pulse detection and power analysis
- **waveform_capture.ino** - Used for waveform capture and analysis

Upload the appropriate sketch to your ESP32 depending on the type of analysis you want to perform:

1. Open the Arduino IDE
2. Open either `pulse_detector/pulse_detector.ino` or `waveform_capture/waveform_capture.ino`
3. Select your ESP32 board and port in the Arduino IDE
4. Click Upload

### 2. Run Python Analysis Scripts

Three Python scripts are provided for different types of analysis:

- **pulse_detector.py** - For pulse timing analysis
- **power_analysis.py** - For power consumption analysis
- **waveform_detector.py** - For waveform capture and analysis

#### For Pulse Detection:

1. Upload `pulse_detector.ino` to your ESP32
2. Connect your ESP32 to your computer
3. Run: `python pulse_detector.py`
4. Follow the prompts to:
   - Select the serial port
   - Enter a session name (or use the default with prefix "pulse_")
   - Choose the data collection duration
   - View the results

#### For Power Analysis:

1. Upload `pulse_detector.ino` to your ESP32 (same sketch is used)
2. Connect your ESP32 to your computer
3. Run: `python power_analysis.py`
4. Follow the prompts to:
   - Select the serial port
   - Enter a session name (or use the default with prefix "power_")
   - Enter the pulse voltage and known resistor value
   - Choose the data collection duration
   - View the analysis results and plots

#### For Waveform Analysis:

1. Upload `waveform_capture.ino` to your ESP32
2. Connect your ESP32 to your computer
3. Run: `python waveform_detector.py`
4. Follow the prompts to:
   - Select the serial port
   - Enter a session name (or use the default with prefix "waveform_")
   - Choose the data collection duration
   - View the captured waveforms and analysis

## Data Output

All data is saved in the `data/` directory, organized by session name. Each session folder contains:

- Raw data CSV files
- Analysis text reports
- A `plots/` directory with visualization images

Example folder structure:
```
data/
    pulse_session_2025-05-09_1442/
        pulse_analysis.csv
        pulse_analysis.txt
        pulse_data_corrupted.csv
        pulse_data_raw.csv
        pulse_plots_index.txt
        plots/
            pulse_interval_hist.png
            pulse_intervals.png
            pulse_signal.png
            pulse_voltage_hist.png
    
    power_session_2025-05-09_1440/
        power_analysis.txt
        power_data_corrupted.csv
        power_data_raw.csv
        power_plots_index.txt
        plots/
            power_current_hist.png
            power_current.png
            power_distributions.png
            power_power_hist.png
            power_power.png
            power_resistance_hist.png
            power_resistance.png
            power_voltage_hist.png
            power_voltage.png
            
    waveform_session_2025-05-09_1445/
        waveform_analysis.csv
        waveform_analysis.txt
        waveform_data_corrupted.csv
        waveform_data_raw.csv
        waveform_plots_index.txt
        plots/
            waveform_all_pulses.png
            waveform_samples_hist.png
            waveform_timing_hist.png
            waveform_voltage_hist.png
            waveform_width_hist.png
```

## Analysis Features

### Pulse Detection
- Measures pulse timing and intervals
- Generates histograms of pulse intervals
- Analyzes pulse voltage characteristics

### Power Analysis
- Measures internal resistance
- Calculates power consumption
- Analyzes current flow
- Generates time-series plots and distribution histograms

### Waveform Capture
- Captures detailed waveform shape
- Analyzes pulse width and timing
- Provides statistical analysis of waveform characteristics

## Troubleshooting

If you encounter issues:

1. **No serial port detected** - Check your USB connection and drivers


## Notes

- Each script will automatically create folders with the appropriate prefix (pulse_, power_, or waveform_)
- Analysis and visualization files are automatically generated based on the collected data
- For best results, use the recommended resistor values

