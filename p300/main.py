import pygame
import random
import time
import csv
import serial
import numpy as np
import pandas as pd
import threading
import os
from datetime import datetime
import sys

class ChessboardFlipperExperiment:
    def __init__(self):
        # Pygame Initialization
        pygame.init()
        
        # Screen dimensions based on the current screen resolution
        self.screen_info = pygame.display.Info()
        self.window_width = self.screen_info.current_w
        self.window_height = self.screen_info.current_h
        
        # Experiment parameters (configurable)
        self.board_size = 6
        self.square_size = min(self.window_width, self.window_height) // 10  # Adapt to screen size
        self.spacing = max(5, self.square_size // 20)  # Adaptive spacing
        self.flip_count_target = 50  # Number of flips to perform
        self.min_interval = 1.0  # Minimum time between flips (seconds)
        self.max_interval = 3.0  # Maximum time between flips (seconds)
        self.flash_duration = 0.5  # How long to show the flipped board (seconds)
        self.rest_duration = 0.8  # How long to show the original board after flip (seconds)
        
        # Define colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (50, 50, 50)
        self.START_COLOR = (70, 120, 200)  # Blue for the start button
        self.TEXT_COLOR = (255, 255, 255)  # White text
        self.STATUS_COLOR = (50, 200, 70)  # Green for status indicators
        self.WARNING_COLOR = (200, 50, 50)  # Red for warnings
        
        # Calculate board position
        self.board_width = self.board_size * self.square_size + (self.board_size - 1) * self.spacing
        self.board_height = self.board_size * self.square_size + (self.board_size - 1) * self.spacing
        self.x_offset = (self.window_width - self.board_width) // 2
        self.y_offset = (self.window_height - self.board_height) // 2
        
        # Font setup
        self.title_font = pygame.font.SysFont('Arial', 48, bold=True)
        self.button_font = pygame.font.SysFont('Arial', 36)
        self.status_font = pygame.font.SysFont('Arial', 24)
        self.timer_font = pygame.font.SysFont('Arial', 72, bold=True)
        
        # Experiment data
        self.trigger_times = []
        self.combined_data = []
        
        # Set up the Pygame screen
        self.screen = pygame.display.set_mode((self.window_width, self.window_height),pygame.FULLSCREEN)
        pygame.display.set_caption('Chessboard Flipper for P300')
        
        # Serial communication settings
        self.arduino_port = 'COM4'  # Default port, can be changed via UI
        self.baud_rate = 115200
        self.ser = None
        self.serial_connected = False
        
        # Experiment state
        self.running = False
        self.paused = False
        self.completed = False
        self.participant_id = "P001"  # Default participant ID
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.start_time = 0
        self.current_flip = 0
        self.stop_event = threading.Event()
        self.eeg_thread = None
        
        # Clock for timing
        self.clock = pygame.time.Clock()

    def connect_serial(self):
        """Attempt to connect to the Arduino serial port"""
        try:
            self.ser = serial.Serial(self.arduino_port, self.baud_rate, timeout=1)
            time.sleep(2)  # Wait for Arduino to initialize
            self.serial_connected = True
            return True
        except serial.SerialException as e:
            print(f"Serial connection error: {e}")
            self.serial_connected = False
            return False

    def draw_chessboard(self):
        """Draw the original chessboard pattern"""
        for row in range(self.board_size):
            for col in range(self.board_size):
                color = self.WHITE if (row + col) % 2 == 0 else self.BLACK
                pygame.draw.rect(self.screen, color, (
                    self.x_offset + col * (self.square_size + self.spacing),
                    self.y_offset + row * (self.square_size + self.spacing),
                    self.square_size, self.square_size
                ))

    def flip_chessboard(self):
        """Flip the chessboard and record the flip times"""
        # Flip the colors
        for row in range(self.board_size):
            for col in range(self.board_size):
                color = self.BLACK if (row + col) % 2 == 0 else self.WHITE
                pygame.draw.rect(self.screen, color, (
                    self.x_offset + col * (self.square_size + self.spacing),
                    self.y_offset + row * (self.square_size + self.spacing),
                    self.square_size, self.square_size
                ))
        
        # Draw status info
        self.draw_status_info()
        
        # Update the display
        pygame.display.flip()
        
        # Record the flip time
        flip_time = time.perf_counter() - self.start_time
        self.trigger_times.append(flip_time)
        print(f"Flip {self.current_flip + 1}/{self.flip_count_target} recorded at time: {flip_time:.4f}s")
        
        # Wait for flash duration
        time.sleep(self.flash_duration)
        
        # Draw original chessboard
        self.draw_chessboard()
        self.draw_status_info()
        pygame.display.flip()
        
        # Increment flip counter
        self.current_flip += 1

    def draw_status_info(self):
        """Draw status information on the screen"""
        # Progress bar
        progress_width = int((self.current_flip / self.flip_count_target) * self.window_width * 0.8)
        progress_height = 20
        progress_x = self.window_width * 0.1
        progress_y = self.window_height - 50
        
        # Border for progress bar
        pygame.draw.rect(self.screen, self.GRAY, (
            progress_x - 2, progress_y - 2,
            self.window_width * 0.8 + 4, progress_height + 4
        ))
        
        # Progress fill
        pygame.draw.rect(self.screen, self.STATUS_COLOR, (
            progress_x, progress_y,
            progress_width, progress_height
        ))
        
        # Status text
        status_text = f"Participant: {self.participant_id} | Flip: {self.current_flip}/{self.flip_count_target}"
        if self.serial_connected:
            status_text += " | EEG: Connected"
        else:
            status_text += " | EEG: Not Connected"
        
        status_surface = self.status_font.render(status_text, True, self.TEXT_COLOR)
        self.screen.blit(status_surface, (self.window_width * 0.1, progress_y - 30))
        
        # Draw experiment elapsed time
        if self.start_time > 0:
            elapsed = time.perf_counter() - self.start_time
            time_text = f"{elapsed:.1f}s"
            time_surface = self.status_font.render(time_text, True, self.TEXT_COLOR)
            self.screen.blit(time_surface, (self.window_width * 0.9 - time_surface.get_width(), 20))

    def show_start_screen(self):
        """Display the start screen with experiment configuration options"""
        self.screen.fill(self.BLACK)
        
        # Title
        title_text = self.title_font.render("P300 Chessboard Flipper Experiment", True, self.TEXT_COLOR)
        self.screen.blit(title_text, (self.window_width // 2 - title_text.get_width() // 2, 50))
        
        # Participant ID input
        id_text = self.button_font.render(f"Participant ID: {self.participant_id}", True, self.TEXT_COLOR)
        self.screen.blit(id_text, (self.window_width // 2 - id_text.get_width() // 2, 150))
        
        # Serial port status
        port_status = "Connected" if self.serial_connected else "Not Connected"
        port_color = self.STATUS_COLOR if self.serial_connected else self.WARNING_COLOR
        port_text = self.button_font.render(f"EEG Device: {port_status} (Port: {self.arduino_port})", True, port_color)
        self.screen.blit(port_text, (self.window_width // 2 - port_text.get_width() // 2, 220))
        
        # Connect button if not connected
        if not self.serial_connected:
            connect_rect = pygame.Rect(self.window_width // 2 - 150, 270, 300, 60)
            pygame.draw.rect(self.screen, self.START_COLOR, connect_rect)
            connect_text = self.button_font.render("Connect EEG Device", True, self.TEXT_COLOR)
            self.screen.blit(connect_text, (connect_rect.centerx - connect_text.get_width() // 2, connect_rect.centery - connect_text.get_height() // 2))
        
        # Experiment configuration
        config_text = self.button_font.render(f"Flips: {self.flip_count_target} | Interval: {self.min_interval:.1f}s-{self.max_interval:.1f}s", True, self.TEXT_COLOR)
        self.screen.blit(config_text, (self.window_width // 2 - config_text.get_width() // 2, 350))
        
        # Draw the start button
        start_rect = pygame.Rect(self.window_width // 2 - 150, 450, 300, 80)
        pygame.draw.rect(self.screen, self.START_COLOR, start_rect)
        start_text = self.button_font.render("Start Experiment", True, self.TEXT_COLOR)
        self.screen.blit(start_text, (start_rect.centerx - start_text.get_width() // 2, start_rect.centery - start_text.get_height() // 2))
        
        # Instructions
        instructions = [
            "Instructions:",
            "1. Make sure the EEG device is connected",
            "2. Enter participant information",
            "3. Press 'Start Experiment' to begin",
            "4. Press ESC at any time to quit"
        ]
        
        for i, line in enumerate(instructions):
            inst_text = self.status_font.render(line, True, self.TEXT_COLOR)
            self.screen.blit(inst_text, (self.window_width // 2 - 200, 550 + i * 30))
        
        pygame.display.flip()
        
        # Handle events on the start screen
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.KEYDOWN:
                    # Allow changing participant ID with keyboard
                    if event.key == pygame.K_BACKSPACE:
                        self.participant_id = self.participant_id[:-1]
                    elif event.key == pygame.K_RETURN:
                        waiting = False
                    elif event.unicode.isalnum() or event.unicode == '_':
                        self.participant_id += event.unicode
                    self.show_start_screen()  # Refresh the screen
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    # Check if start button was clicked
                    if start_rect.collidepoint(event.pos):
                        waiting = False
                    # Check if connect button was clicked
                    elif not self.serial_connected and connect_rect.collidepoint(event.pos):
                        if self.connect_serial():
                            self.show_start_screen()  # Refresh with connected status

    def show_countdown(self):
        """Show a countdown before starting the experiment"""
        for i in range(3, 0, -1):
            self.screen.fill(self.BLACK)
            count_text = self.timer_font.render(str(i), True, self.TEXT_COLOR)
            self.screen.blit(count_text, (self.window_width // 2 - count_text.get_width() // 2, 
                                         self.window_height // 2 - count_text.get_height() // 2))
            pygame.display.flip()
            time.sleep(1)
        
        # Get ready text
        self.screen.fill(self.BLACK)
        ready_text = self.timer_font.render("Get Ready!", True, self.TEXT_COLOR)
        self.screen.blit(ready_text, (self.window_width // 2 - ready_text.get_width() // 2, 
                                     self.window_height // 2 - ready_text.get_height() // 2))
        pygame.display.flip()
        time.sleep(1)

    def show_completion_screen(self):
        """Show experiment completion screen"""
        self.screen.fill(self.BLACK)
        
        # Title
        complete_text = self.title_font.render("Experiment Complete!", True, self.STATUS_COLOR)
        self.screen.blit(complete_text, (self.window_width // 2 - complete_text.get_width() // 2, 100))
        
        # Stats
        stats = [
            f"Participant: {self.participant_id}",
            f"Total Flips: {self.current_flip}",
            f"Total Samples: {len(self.combined_data)}",
            f"Data saved to: ./data/{self.participant_id}_{self.session_id}/"
        ]
        
        for i, line in enumerate(stats):
            stat_text = self.button_font.render(line, True, self.TEXT_COLOR)
            self.screen.blit(stat_text, (self.window_width // 2 - stat_text.get_width() // 2, 200 + i * 50))
        
        # Exit instructions
        exit_text = self.status_font.render("Press ESC to exit", True, self.TEXT_COLOR)
        self.screen.blit(exit_text, (self.window_width // 2 - exit_text.get_width() // 2, 
                                    self.window_height - 100))
        
        pygame.display.flip()
        
        # Wait for exit
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    waiting = False

    def collect_eeg_data(self):
        """Thread function to collect EEG data"""
        trigger_index = 0
        eeg_sample_number = 0
        
        while not self.stop_event.is_set():
            if self.serial_connected and self.ser.in_waiting > 0:
                try:
                    raw_data = self.ser.readline().decode().strip()  # Properly decode the data
                    eeg_signal = int(raw_data)  # Convert the data to an integer
                    current_time = time.perf_counter() - self.start_time  # Calculate elapsed time
                    
                    # Time window for trigger detection
                    time_window = 0.050  # 50 milliseconds window
                    
                    # Check if this sample is near a trigger time
                    is_trigger = False
                    if trigger_index < len(self.trigger_times):
                        flip_time = self.trigger_times[trigger_index]
                        
                        if abs(flip_time - current_time) < time_window:
                            is_trigger = True
                            trigger_index += 1
                    
                    # Store the data
                    self.combined_data.append([
                        eeg_sample_number, 
                        current_time, 
                        eeg_signal, 
                        'Trigger' if is_trigger else ''
                    ])
                    
                    eeg_sample_number += 1
                    
                except (ValueError, UnicodeDecodeError) as e:
                    print(f"Error processing EEG data: {e}")
                    print(f"Raw data: {raw_data}")
            
            # Sleep to avoid busy waiting
            time.sleep(0.001)

    def save_experiment_data(self):
        """Save all experimental data to CSV files"""
        # Create a directory structure for the data
        data_dir = f"data/{self.participant_id}_{self.session_id}"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save EEG and trigger data
        if self.combined_data:
            print(f"Writing {len(self.combined_data)} samples to CSV.")
            df = pd.DataFrame(
                self.combined_data, 
                columns=["EEG Sample Number", "EEG Sample Time (s)", "EEG Signal", "Trigger"]
            )
            df.to_csv(f"{data_dir}/combined_eeg_trigger_data.csv", index=False)
        
        # Save trigger times
        with open(f"{data_dir}/trigger_times.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Trial", "Flip", "Time (s)"])
            for i, flip_time in enumerate(self.trigger_times):
                writer.writerow([1, i + 1, flip_time])
        
        # Save experiment metadata
        metadata = {
            "participant_id": self.participant_id,
            "session_id": self.session_id,
            "date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "board_size": self.board_size,
            "total_flips": self.current_flip,
            "flip_interval_min": self.min_interval,
            "flip_interval_max": self.max_interval,
            "flash_duration": self.flash_duration,
            "total_samples": len(self.combined_data),
            "serial_connected": self.serial_connected,
            "arduino_port": self.arduino_port
        }
        
        with open(f"{data_dir}/experiment_metadata.csv", 'w', newline='') as file:
            writer = csv.writer(file)
            for key, value in metadata.items():
                writer.writerow([key, value])

    def run_experiment(self):
        """Main function to run the experiment"""
        # Show start screen and get experiment settings
        self.show_start_screen()
        
        # Ensure serial connection is established if possible
        if not self.serial_connected and not self.connect_serial():
            print("Warning: Could not connect to EEG device. Running in simulation mode.")
        
        # Show countdown
        self.show_countdown()
        
        # Initialize experiment variables
        self.current_flip = 0
        self.trigger_times = []
        self.combined_data = []
        self.running = True
        self.completed = False
        self.stop_event.clear()
        
        # Draw initial chessboard
        self.screen.fill(self.BLACK)
        self.draw_chessboard()
        self.draw_status_info()
        pygame.display.flip()
        
        # Start timing
        self.start_time = time.perf_counter()
        
        # Start EEG data collection thread
        if self.serial_connected:
            self.eeg_thread = threading.Thread(
                target=self.collect_eeg_data
            )
            self.eeg_thread.daemon = True  # Thread will exit when main program exits
            self.eeg_thread.start()
        
        # Main experiment loop
        next_flip_time = time.perf_counter() + random.uniform(self.min_interval, self.max_interval)
        
        while self.running and self.current_flip < self.flip_count_target:
            current_time = time.perf_counter()
            
            # Check if it's time for a flip
            if current_time >= next_flip_time:
                self.flip_chessboard()
                
                # Calculate next flip time
                next_flip_time = current_time + self.rest_duration + random.uniform(self.min_interval, self.max_interval)
            
            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    self.running = False
                    break
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    # Toggle pause on spacebar
                    self.paused = not self.paused
            
            # Update screen if not flipping
            if current_time < next_flip_time:
                self.draw_status_info()
                pygame.display.flip()
            
            # Cap the frame rate
            self.clock.tick(60)
        
        # Mark experiment as completed if we finished all flips
        self.completed = self.current_flip >= self.flip_count_target
        
        # Stop the EEG collection thread
        self.stop_event.set()
        if self.eeg_thread:
            self.eeg_thread.join(timeout=2)
        
        # Save all data
        self.save_experiment_data()
        
        # Show completion screen
        if self.completed:
            self.show_completion_screen()
        
        # Clean up
        if self.serial_connected and self.ser:
            self.ser.close()
        
        # Quit pygame
        pygame.quit()

# Create and run the experiment
if __name__ == "__main__":
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Run the experiment
    experiment = ChessboardFlipperExperiment()
    experiment.run_experiment()