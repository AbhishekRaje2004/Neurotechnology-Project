import random
import time
from psychopy import visual, core, event

def run_oddball_task(win, ser, trials=100, live_visualization=None):
    """
    Run a standard P300 oddball paradigm
    
    Parameters:
    -----------
    win : psychopy.visual.Window
        PsychoPy window to draw stimuli on
    ser : serial.Serial
        Serial connection to Arduino
    trials : int
        Number of trials to run (total stimuli presentations)
    live_visualization : LiveEEGPlotter or None
        Live visualization interface (if available)
        
    Returns:
    --------
    event_log : list
        List of event timestamps and types [(timestamp, event_type), ...]
    """
    # Create stimuli
    stim_freq = visual.Circle(win, radius=0.1, fillColor='blue')
    stim_rare = visual.Rect(win, width=0.2, height=0.2, fillColor='red')
    
    # Create stimulus sequence with appropriate target/non-target ratio
    # P300 is best detected with rare targets (15-20% of stimuli)
    target_ratio = 0.2  # 20% targets
    stimuli = [0]*int(trials*(1-target_ratio)) + [1]*int(trials*target_ratio)
    random.shuffle(stimuli)
    
    # Display instructions
    instr = visual.TextStim(win, text="Focus on the red squares and mentally count them.\n\n"
                           "Press any key to start.", color='white')
    instr.draw()
    win.flip()
    event.waitKeys()
    
    # Run the task
    clock = core.Clock()
    event_log = []
    
    # Counters for feedback
    count_standard = 0
    count_target = 0
    last_10_responses = []
    
    for i, stim_type in enumerate(stimuli):
        # Inter-stimulus interval (jittered to avoid expectation)
        win.flip()  # Clear screen
        core.wait(random.uniform(0.5, 1.0))
        
        # Draw the stimulus
        (stim_freq if stim_type == 0 else stim_rare).draw()
        
        # Progress indicator
        progress = visual.TextStim(win, text=f"{i+1}/{trials}",
                                 pos=[0.9, 0.9], color='gray', height=0.03)
        progress.draw()
        
        # Display stimulus and send trigger
        win.flip()
        
        # Send trigger to Arduino
        ser.write(f"{stim_type}\n".encode())
        
        # Log the event
        ts = time.time()
        event_log.append((ts, stim_type))
        
        # Update counters
        if stim_type == 0:
            count_standard += 1
        else:
            count_target += 1
            
        # If live visualization is active, update it
        if live_visualization is not None:
            try:
                # Get latest reading from serial and add to visualization
                line = ser.readline().decode('utf-8').strip()
                parts = line.split(',')
                if len(parts) == 3:
                    eeg_value = float(parts[1])
                    live_visualization.add_data_point(ts, eeg_value, stim_type)
            except:
                pass  # Ignore errors in visualization updates
        
        # Show the stimulus for a fixed duration
        stimulus_duration = 0.2  # 200ms, standard for P300 oddball
        core.wait(stimulus_duration)
        
        # Check for escape key to exit early
        if event.getKeys(['escape']):
            break
    
    # Show completion message
    completion = visual.TextStim(win, text=f"Task complete!\n\n"
                               f"Standard stimuli: {count_standard}\n"
                               f"Target stimuli: {count_target}\n\n"
                               f"Press any key to continue.", color='white')
    completion.draw()
    win.flip()
    event.waitKeys()
    
    return event_log


def run_p300_calibration(win, ser, trials_per_character=10, target_chars=None, live_visualization=None, speller=None):
    """
    Run a P300 speller calibration sequence for a set of target characters
    
    Parameters:
    -----------
    win : psychopy.visual.Window
        PsychoPy window to draw stimuli on
    ser : serial.Serial
        Serial connection to Arduino
    trials_per_character : int
        Number of complete flash sequences per character
    target_chars : list or None
        List of target characters to use for calibration
    live_visualization : LiveEEGPlotter or None
        Live visualization interface (if available)
    speller : P300Speller or None
        Existing speller instance (if None, a new one will be created)
        
    Returns:
    --------
    calibration_data : dict
        Dictionary with calibration data
    """
    from speller import P300Speller
    
    # Use existing speller or create a new one if not provided
    if speller is None:
        # Create speller interface - use the existing window (already fullscreen from main.py)
        speller = P300Speller(win, ser)
    
    # Default target characters if none provided
    if target_chars is None:
        target_chars = ['A', 'B', 'P', '7']
    
    # Show instructions with larger font for better visibility in fullscreen
    instr = visual.TextStim(win, text="P300 Calibration\n\n"
                          "You will be shown a sequence of characters to focus on.\n"
                          "Count each time your target character flashes.\n\n"
                          "Press any key to begin.", color='white', height=0.08)  # Increased text size
    instr.draw()
    win.flip()
    event.waitKeys()
    
    all_events = []
    all_targets = []
    
    # Run calibration for each character
    for target in target_chars:
        # Highlight target with larger font
        target_text = visual.TextStim(win, text=f"Focus on the character: {target}\n\n"
                                    "Press any key when ready.", color='white', height=0.08)  # Increased text size
        target_text.draw()
        win.flip()
        event.waitKeys()
        
        # Run calibration sequence
        events, target_pos = speller.run_calibration(target, n_sequences=trials_per_character)
        
        # Store data
        all_events.extend(events)
        all_targets.append((target, target_pos))
        
        # Provide feedback with larger font
        feedback = visual.TextStim(win, text=f"Completed {target}.\n\n"
                                 f"{len(target_chars) - target_chars.index(target) - 1} characters remaining.\n\n"
                                 "Press any key to continue.", color='white', height=0.08)  # Increased text size
        feedback.draw()
        win.flip()
        event.waitKeys()
    
    # Completion message with larger font
    completion = visual.TextStim(win, text="Calibration complete!\n\n"
                               "Press any key to continue.", color='white', height=0.08)  # Increased text size
    completion.draw()
    win.flip()
    event.waitKeys()
    
    # Return calibration data
    calibration_data = {
        'events': all_events,
        'targets': all_targets,
        'n_sequences': trials_per_character
    }
    
    return calibration_data
