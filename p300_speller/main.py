from psychopy import visual, core, event
import serial
import numpy as np
import argparse
import time
import os
import tkinter as tk
import sys

# Import custom modules
from acquisition_arduino import ArduinoEEGAcquisition
from epoching import extract_epochs
from preprocessing import preprocess_eeg
from analysis import ensemble_average, plot_erp
from p300_classifier import P300Classifier
from speller import P300Speller
from stimulus import run_oddball_task, run_p300_calibration
from visualization import LiveEEGPlotter, EpochViewer, save_experiment_data, load_experiment_data

# Path to save/load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "data", "p300_model.pkl")

def run_calibration(port='COM4', visualize=True):
    """Run P300 calibration session with visualization"""
    print("Starting P300 calibration session...")
    
    # Initialize components
    acq = ArduinoEEGAcquisition(port=port)
    if not acq.connect():
        print("Failed to connect to Arduino")
        return False
    
    # Create PsychoPy window - windowed mode with white background
    win = visual.Window(size=(800, 600), color='white', fullscr=False)
    
    # Create P300 classifier
    classifier = P300Classifier()
    
    # Initialize live visualization if requested
    live_viz = None
    if visualize:
        # Create a Tkinter root window
        tk_root = tk.Tk()
        tk_root.withdraw()  # Hide the root window
        
        # Initialize the live EEG plotter
        live_viz = LiveEEGPlotter()
        viz_window = live_viz.initialize("P300 Calibration - Live Signal")
    
    # Start EEG acquisition
    acq.start_acquisition()
    
    # Define calibration targets (balanced across matrix)
    calibration_targets = ['A', 'D', 'G', 'J', 'M', 'P', 'S', 'V', 'Y', '2', '5', '8']
    n_sequences = 15  # More sequences for good training data
    
    # Create speller instance
    speller = P300Speller(win, acq.ser, classifier)
    
    # Calibration intro text with larger font (black for white background)
    intro = visual.TextStim(win, text="P300 Calibration\n\nYou will be shown a sequence of characters to focus on.\n"
                           "When a character is highlighted, focus on it and count each time it flashes.\n\n"
                           "Press any key to begin\nPress ESC at any time to exit", 
                           color='black', height=0.08)  # Increased text size, black color
    intro.draw()
    win.flip()
    keys = event.waitKeys()
    if 'escape' in keys:
        acq.stop_acquisition()
        acq.disconnect()
        win.close()
        return False
    
    # Run calibration for each target
    all_epochs = []
    all_labels = []
    
    try:
        for target in calibration_targets:
            # Show target instruction with larger font (black for white background)
            instr = visual.TextStim(win, text=f"Focus on the letter '{target}'\nand mentally count each time it flashes.\n\n"
                                  "Press any key when ready\nPress ESC to exit", 
                                  color='black', height=0.08)  # Black text
            instr.draw()
            win.flip()
            keys = event.waitKeys()
            if 'escape' in keys:
                raise KeyboardInterrupt
            
            # Clear all previous EEG data
            acq.clear_data()
            
            # Run calibration sequence
            flash_events, target_pos = speller.run_calibration(target, n_sequences=n_sequences)
            
            # Get collected data
            timestamps, eeg_data, triggers = acq.get_data()
            
            # Update live visualization if active
            if visualize and live_viz:
                for i, (ts, value) in enumerate(zip(timestamps, eeg_data)):
                    marker = triggers[i] if i < len(triggers) and triggers[i] is not None else None
                    live_viz.add_data_point(ts, value[0], marker)
                    
            # Apply preprocessing to the raw EEG data
            eeg_data = np.array(eeg_data)
            if eeg_data.shape[0] > 0:
                eeg_data = preprocess_eeg(eeg_data.T).T  # The EEG data is stored as (samples, channels)
            
            # Process the data
            epochs, labels = extract_epochs(eeg_data, timestamps, flash_events)
            
            # Store data for training
            all_epochs.extend(epochs)
            all_labels.extend(labels)
            
            # Short break (black text for white background)
            rest = visual.TextStim(win, text="Good job! Take a short break.\n\nPress any key when ready for the next target\nPress ESC to exit", 
                                color='black', height=0.08)  # Black text
            rest.draw()
            win.flip()
            keys = event.waitKeys()
            if 'escape' in keys:
                raise KeyboardInterrupt
    
    except KeyboardInterrupt:
        print("Calibration interrupted by user")
        acq.stop_acquisition()
        acq.disconnect()
        win.close()
        return False

    # Convert to numpy arrays
    all_epochs = np.array(all_epochs)
    all_labels = np.array(all_labels)
    
    # Train the classifier
    print("Training P300 classifier...")
    accuracy, cm = classifier.train(all_epochs, all_labels)
    
    # Print detailed calibration metrics
    print("\n===== P300 CALIBRATION RESULTS =====")
    print(f"Total epochs: {len(all_epochs)} ({sum(all_labels)} targets, {len(all_labels) - sum(all_labels)} non-targets)")
    print(f"Classifier accuracy: {accuracy:.4f}")
    
    # Calculate and print more detailed metrics
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    y_pred = classifier.pipeline.predict(classifier.extract_features(all_epochs))
    y_proba = classifier.pipeline.predict_proba(classifier.extract_features(all_epochs))[:, 1]
    
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(all_labels, y_pred)
    recall = recall_score(all_labels, y_pred)
    f1 = f1_score(all_labels, y_pred)
    
    try:
        auc = roc_auc_score(all_labels, y_proba)
        print(f"ROC AUC: {auc:.4f}")
    except:
        print("ROC AUC: Could not calculate")
    
    print(f"Sensitivity (recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Confusion matrix:\n{cm}")
    print("===================================")
    
    # Plot average ERPs
    target_idx = [i for i, l in enumerate(all_labels) if l == 1]
    non_target_idx = [i for i, l in enumerate(all_labels) if l == 0]
    
    target_avg = ensemble_average(all_epochs, target_idx)
    non_target_avg = ensemble_average(all_epochs, non_target_idx)
    
    # Plot both target and non-target averages
    plot_erp_comparison(target_avg, non_target_avg)
    
    # Clean up
    acq.stop_acquisition()
    acq.disconnect()
    
    # Save experiment data
    save_experiment_data(
        all_epochs, all_labels,
        timestamps=timestamps,
        eeg_data=eeg_data,
        events=flash_events,
        filename="calibration_data.pkl"
    )
    
    # Save classifier
    try:
        classifier.save_model(MODEL_PATH)
        print(f"P300 classifier model saved to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model: {e}")
    
    completed = visual.TextStim(win, text="Calibration complete!\n\nPress any key to exit", 
                              color='black', height=0.08)  # Black text for white background
    completed.draw()
    win.flip()
    event.waitKeys()
    win.close()
    
    # Show the epoch viewer if visualization is enabled
    if visualize:
        epoch_viewer = EpochViewer()
        epoch_viewer.show(all_epochs, all_labels)
        tk_root.mainloop()  # Start the Tkinter event loop
        
    return classifier

def plot_erp_comparison(target_avg, non_target_avg, fs=250):
    """Plot comparison of target and non-target ERPs"""
    import matplotlib.pyplot as plt
    
    timepoints = target_avg.shape[1]
    t = np.linspace(-0.2, 0.8, timepoints)
    
    plt.figure(figsize=(12, 6))
    plt.plot(t, target_avg[0], 'b-', linewidth=2, label='Target (P300)')
    plt.plot(t, non_target_avg[0], 'r-', linewidth=2, label='Non-target')
    plt.axvline(0, color='k', linestyle='--', label='Stimulus')
    plt.axvspan(0.25, 0.5, color='yellow', alpha=0.2, label='P300 window')
    plt.title('P300 Response - Target vs Non-target')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.grid(True)
    plt.legend()
    plt.show()

def run_speller(classifier, port='COM4', visualize=True):
    """Run the P300 speller with a trained classifier and visualization"""
    # Initialize Tkinter for visualization if needed
    tk_root = None
    live_viz = None
    epoch_viewer = None
    
    if visualize:
        tk_root = tk.Tk()
        tk_root.withdraw()
        live_viz = LiveEEGPlotter()
        viz_window = live_viz.initialize("P300 Speller - Live Signal")
    
    # Load model if not provided
    if classifier is None:
        if os.path.exists(MODEL_PATH):
            try:
                classifier = P300Classifier()
                classifier.load_model(MODEL_PATH)
            except Exception as e:
                print(f"Error loading model: {e}")
                return
        else:
            print(f"No trained classifier found at {MODEL_PATH}")
            print("Please run calibration first.")
            return
    
    if not classifier.trained:
        print("Error: No trained classifier available")
        return
    
    print("Starting P300 speller session...")
    
    # Initialize components
    acq = ArduinoEEGAcquisition(port=port)
    if not acq.connect():
        print("Failed to connect to Arduino")
        return
    
    # Create PsychoPy window - windowed mode with white background
    win = visual.Window(size=(800, 600), color='white', fullscr=False)
    
    # Create speller interface
    speller = P300Speller(win, acq.ser, classifier)
    
    # Start EEG acquisition
    acq.start_acquisition()
    
    # Speller intro text (black text for white background)
    intro = visual.TextStim(win, text="P300 Speller\n\nFocus on the character you want to type\n"
                           "and mentally count each time it flashes.\n\n"
                           "Press any key to begin\nPress Escape at any time to exit", 
                           color='black', height=0.05)
    intro.draw()
    win.flip()
    event.waitKeys()
    
    # Number of sequences for each character
    n_sequences = 10
    running = True
    
    # Store all epochs for post-experiment review
    all_trial_epochs = []
    all_trial_labels = []
    all_selected_chars = []
    
    while running:
        # Draw the speller matrix
        speller.draw_matrix()
        
        # Wait for keypress to start typing or exit
        keys = event.waitKeys()
        if 'escape' in keys:
            running = False
            break
        
        # Clear previous data
        acq.clear_data()
        
        # Run typing sequence
        flash_events = speller.run_typing_sequence(n_sequences=n_sequences)
        
        # Get collected data
        timestamps, eeg_data, triggers = acq.get_data()
        
        # Update live visualization if active
        if visualize and live_viz:
            for i, (ts, value) in enumerate(zip(timestamps, eeg_data)):
                marker = triggers[i] if i < len(triggers) and triggers[i] is not None else None
                live_viz.add_data_point(ts, value[0], marker)
        
        # Apply preprocessing to the raw EEG data
        eeg_data = np.array(eeg_data)
        if eeg_data.shape[0] > 0:
            eeg_data = preprocess_eeg(eeg_data.T).T
        
        # Process the data
        epochs, _ = extract_epochs(eeg_data, timestamps, flash_events)
        
        if len(epochs) == 0:
            print("No valid epochs found")
            continue
            
        # Detect selected character
        try:
            selected_char, confidence = speller.detect_p300(epochs, flash_events, None)
            
            # Store epochs for post-experiment review
            # Create labels using detected character
            target_rc = speller.char_positions[selected_char]
            row, col = target_rc
            
            trial_labels = []
            for i, (_, item) in enumerate(flash_events):
                # Check if this flash contains the selected character
                is_target = False
                if isinstance(item, int):  # Row/column mode
                    is_target = (item == row) or (item == col + 6)
                else:  # Single character mode
                    is_target = (item[0] == row and item[1] == col)
                    
                trial_labels.append(1 if is_target else 0)
            
            all_trial_epochs.extend(epochs)
            all_trial_labels.extend(trial_labels)
            all_selected_chars.append(selected_char)
            
            # Add the detected character
            speller.add_character(selected_char)
            
            # Show feedback (dark green for visibility on white background)
            feedback = visual.TextStim(win, 
                                     text=f"Selected: {selected_char} (confidence: {confidence:.2f})",
                                     pos=[0, -0.8], color='darkgreen', height=0.05)
            feedback.draw()
            speller.text_display.draw()
            win.flip()
            core.wait(1.5)
            
        except Exception as e:
            print(f"Error during character detection: {e}")
    
    # Save all trial data for later review
    if all_trial_epochs:
        save_experiment_data(
            np.array(all_trial_epochs),
            np.array(all_trial_labels),
            events=flash_events,
            filename=f"speller_session_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        )
    
    # Clean up
    acq.stop_acquisition()
    acq.disconnect()
    win.close()
    
    # Show epoch viewer with recorded data
    if visualize and len(all_trial_epochs) > 0:
        epoch_viewer = EpochViewer()
        epoch_viewer.show(np.array(all_trial_epochs), np.array(all_trial_labels))
        
        # Show summary
        print("\nSpelling session results:")
        print(f"Characters typed: {''.join(all_selected_chars)}")
        
        # Start Tkinter main loop to keep visualization windows open
        tk_root.mainloop()

def run_demo(port='COM4', visualize=True):
    """Run a simple P300 oddball paradigm demo with visualization"""
    # Initialize Tkinter for visualization
    tk_root = None
    live_viz = None
    
    if visualize:
        tk_root = tk.Tk()
        tk_root.withdraw()  # Hide the root window
        live_viz = LiveEEGPlotter()
        live_viz.initialize("P300 Oddball Demo - Live Signal")
    
    print("Starting P300 oddball paradigm demo...")
    
    # Initialize components
    acq = ArduinoEEGAcquisition(port=port)
    if not acq.connect():
        print("Failed to connect to Arduino")
        return
    
    # Create PsychoPy window - using windowed mode with white background
    win = visual.Window(size=(800, 600), color='white', fullscr=False)
    
    # Start EEG acquisition
    acq.start_acquisition()
    
    # Show introduction with black text for visibility on white background
    intro = visual.TextStim(win, text="P300 Oddball Demo\n\n"
                          "You will see a series of blue circles (frequent)\n"
                          "and red squares (rare targets).\n\n"
                          "Please mentally count the red squares.\n\n"
                          "Press any key to begin", color='black')
    intro.draw()
    win.flip()
    event.waitKeys()
    
    # Clear any existing data
    acq.clear_data()
    
    # Run oddball task with visualization
    events = run_oddball_task(win, acq.ser, trials=100, live_visualization=live_viz)
    
    # Get collected data
    timestamps, eeg_data, triggers = acq.get_data()
    
    # Apply preprocessing to the raw EEG data
    eeg_data = np.array(eeg_data)
    if eeg_data.shape[0] > 0:
        eeg_data = preprocess_eeg(eeg_data.T).T
    
    # Process the data
    epochs, labels = extract_epochs(eeg_data, timestamps, events)
    
    if len(epochs) == 0:
        print("No valid epochs found")
        return
    
    # ERP averaging
    target_idx = [i for i, l in enumerate(labels) if l == 1]
    non_target_idx = [i for i, l in enumerate(labels) if l == 0]
    
    # Check if we have both target and non-target data
    if len(target_idx) == 0 or len(non_target_idx) == 0:
        print("Not enough target or non-target events were captured")
        return
        
    target_avg = ensemble_average(epochs, target_idx)
    non_target_avg = ensemble_average(epochs, non_target_idx)
    
    # Plot both target and non-target averages
    plot_erp_comparison(target_avg, non_target_avg)
    
    # Save the data for later analysis
    save_experiment_data(
        epochs, 
        labels,
        timestamps=timestamps,
        eeg_data=eeg_data,
        events=events,
        filename=f"oddball_session_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
    )
    
    # Clean up
    acq.stop_acquisition()
    acq.disconnect()
    win.close()
    
    # Show epoch viewer with the data
    if visualize and len(epochs) > 0:
        epoch_viewer = EpochViewer()
        epoch_viewer.show(epochs, labels)
        # Start Tkinter main loop to keep visualization windows open
        tk_root.mainloop()

def view_saved_data():
    """Open a saved data file for visualization"""
    from tkinter import filedialog
    import tkinter as tk
    
    # Create a root window (and hide it)
    root = tk.Tk()
    root.withdraw()
    
    # Open file dialog
    filename = filedialog.askopenfilename(
        title="Select P300 data file to visualize",
        filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
    )
    
    if not filename:
        print("No file selected")
        return
    
    try:
        # Load the data
        data = load_experiment_data(filename)
        print(f"Loaded data from {filename}")
        
        # Extract epochs and labels
        if 'epochs' in data and 'labels' in data:
            epochs = data['epochs']
            labels = data['labels']
            
            # Open the epoch viewer
            viewer = EpochViewer()
            viewer.show(epochs, labels, 
                      fs=data.get('fs', 250),
                      pre=data.get('pre', 0.2),
                      post=data.get('post', 0.8))
            
            root.mainloop()  # Start Tkinter event loop
        else:
            print("Invalid data format - missing epochs or labels")
    
    except Exception as e:
        print(f"Error loading data: {e}")

def verify_visualization_setup():
    """Test function to verify visualization setup"""
    print("Running visualization test...")
    
    # Create a simple Tkinter window
    root = tk.Tk()
    root.title("Visualization Test")
    root.geometry("400x200")
    
    # Add a test label
    label = tk.Label(root, text="If you see this window, Tkinter is working correctly.")
    label.pack(pady=50)
    
    # Add a test button
    def on_click():
        print("Button clicked - GUI is responsive")
    
    button = tk.Button(root, text="Test Button", command=on_click)
    button.pack()
    
    # Schedule the test to run for a few seconds
    root.after(5000, lambda: print("GUI test complete"))
    
    # Start the Tkinter event loop
    root.mainloop()
    
    return True

def run_arduino_visualization_test(port='COM4', duration=30, gain=1.0):
    """Run a direct test connecting to Arduino and visualizing the signal"""
    import serial
    import time
    import tkinter as tk
    from visualization import LiveEEGPlotter
    
    print(f"Running direct visualization test with Arduino on port {port}")
    
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

def main():
    parser = argparse.ArgumentParser(description='P300 Speller System')
    parser.add_argument('--mode', type=str, default='full', 
                      choices=['calibration', 'speller', 'full', 'demo', 'view', 'test', 'viz-test'],
                      help='Operation mode: calibration, speller, full (both), demo, view saved data, test Arduino or test visualization')
    parser.add_argument('--port', type=str, default='COM4', help='Arduino serial port')
    parser.add_argument('--no-viz', action='store_true', help='Disable visualization')
    parser.add_argument('--duration', type=int, default=30, help='Duration for test mode in seconds')
    parser.add_argument('--gain', type=float, default=1.0, help='Gain factor for visualization (amplify signal)')
    
    args = parser.parse_args()
    
    # Test modes
    if args.mode == 'test':
        run_arduino_visualization_test(port=args.port, duration=args.duration)
        return
    elif args.mode == 'viz-test':
        verify_visualization_setup()
        return
    
    # Determine if we should visualize
    visualize = not args.no_viz
    
    if args.mode == 'view':
        view_saved_data()
        return
    
    if args.mode == 'demo':
        run_demo(port=args.port, visualize=visualize)
        return
    
    classifier = None
    
    if args.mode == 'calibration' or args.mode == 'full':
        classifier = run_calibration(port=args.port, visualize=visualize)
    elif args.mode == 'speller':
        # Load classifier from disk
        if os.path.exists(MODEL_PATH):
            try:
                classifier = P300Classifier()
                if classifier.load_model(MODEL_PATH):
                    print("Classifier loaded successfully")
            except Exception as e:
                print(f"Error loading classifier: {e}")
                return
        else:
            print(f"No trained classifier found at {MODEL_PATH}")
            print("Please run calibration first.")
            return
    
    if args.mode == 'speller' or args.mode == 'full':
        if classifier is not None and classifier.trained:
            run_speller(classifier, port=args.port, visualize=visualize)
        else:
            print("Cannot run speller without a trained classifier")

if __name__ == "__main__":
    main()
