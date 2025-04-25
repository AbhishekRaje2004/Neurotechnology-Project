import pygame
import random
import time
import sys
import os
import numpy as np
import serial
import collections

# Initialize pygame
pygame.init()

# Colors and UI settings
BACKGROUND = (240, 245, 250)
WHITE = (255, 255, 255)
BLACK = (40, 40, 40)
LIGHT_GRAY = (220, 225, 230)
GRAY = (180, 185, 190)
YELLOW = (255, 236, 139)
GREEN = (152, 223, 138)
BUTTON_COLOR = (92, 184, 92)
BUTTON_HOVER = (76, 168, 76)
BUTTON_DISABLED = (200, 200, 200)
TITLE_COLOR = (44, 62, 80)
TEXT_BOX_BG = (252, 252, 252)
BORDER_COLOR = (210, 210, 210)

# Screen setup and UI scaling
screen_info = pygame.display.Info()
WIDTH, HEIGHT = screen_info.current_w//2, screen_info.current_h//2
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("P300 Speller - Enhanced UI")

SCALE_FACTOR = min(WIDTH / 1920, HEIGHT / 1080)

# UI Layout Constants
MATRIX_TOP = int(HEIGHT * 0.22)  # Position of the matrix, 22% of screen height
STATUS_TOP = MATRIX_TOP - int(70 * SCALE_FACTOR)  # Position of status display at the top
STATUS_LEFT = WIDTH // 4  # Position for the status display, horizontally centered
STATUS_WIDTH = WIDTH // 2  # Width of the status text area
CELL_SIZE = int(90 * SCALE_FACTOR)  # Size of each cell
CELL_MARGIN = int(10 * SCALE_FACTOR)  # Space between cells
CELL_RADIUS = int(12 * SCALE_FACTOR)  # Rounded corners for cells

TEXT_BOX_TOP = MATRIX_TOP + 6 * (CELL_SIZE + CELL_MARGIN) + int(50 * SCALE_FACTOR)
TEXT_BOX_HEIGHT = int(100 * SCALE_FACTOR)
TEXT_BOX_LEFT = WIDTH // 8
TEXT_BOX_WIDTH = WIDTH * 3 // 4
TEXT_BOX_RADIUS = int(15 * SCALE_FACTOR)
SHADOW_OFFSET = int(4 * SCALE_FACTOR)  # Shadow offset for elements

# Button parameters
BUTTON_WIDTH = int(200 * SCALE_FACTOR)
BUTTON_HEIGHT = int(50 * SCALE_FACTOR)
BUTTON_RADIUS = int(10 * SCALE_FACTOR)
BUTTON_MARGIN = int(20 * SCALE_FACTOR)
BUTTON_TOP = TEXT_BOX_TOP + TEXT_BOX_HEIGHT + BUTTON_MARGIN
BUTTON_LEFT_START = WIDTH // 2 - BUTTON_WIDTH - BUTTON_MARGIN // 2
BUTTON_RIGHT_START = WIDTH // 2 + BUTTON_MARGIN // 2

# Initialize fonts
try:
    # Try to use a nicer font if available
    font_title = pygame.font.Font(None, int(56 * SCALE_FACTOR))
    font_cell = pygame.font.Font(None, int(52 * SCALE_FACTOR))
    font_button = pygame.font.Font(None, int(36 * SCALE_FACTOR))
    font_status = pygame.font.Font(None, int(40 * SCALE_FACTOR))
    font_text = pygame.font.Font(None, int(42 * SCALE_FACTOR))
except:
    # Fall back to system font
    font_title = pygame.font.SysFont('Arial', int(48 * SCALE_FACTOR), bold=True)
    font_cell = pygame.font.SysFont('Arial', int(44 * SCALE_FACTOR), bold=True)
    font_button = pygame.font.SysFont('Arial', int(32 * SCALE_FACTOR))
    font_status = pygame.font.SysFont('Arial', int(36 * SCALE_FACTOR), bold=False)
    font_text = pygame.font.SysFont('Arial', int(38 * SCALE_FACTOR))

# Define matrix
matrix = [
    ['A', 'B', 'C', 'D', 'E', 'F'],
    ['G', 'H', 'I', 'J', 'K', 'L'],
    ['M', 'N', 'O', 'P', 'Q', 'R'],
    ['S', 'T', 'U', 'V', 'W', 'X'],
    ['Y', 'Z', '1', '2', '3', '4'],
    ['5', '6', '7', '8', '9', '_']
]

# Initialize cells with positions and properties
cells = []
for row_idx in range(6):
    cell_row = []
    for col_idx in range(6):
        left = WIDTH // 2 - (3 * CELL_SIZE + 2.5 * CELL_MARGIN) + col_idx * (CELL_SIZE + CELL_MARGIN)
        top = MATRIX_TOP + row_idx * (CELL_SIZE + CELL_MARGIN)
        rect = pygame.Rect(left, top, CELL_SIZE, CELL_SIZE)
        cell = {
            "char": matrix[row_idx][col_idx],
            "rect": rect,
            "flash": False,
            "selected": False
        }
        cell_row.append(cell)
    cells.append(cell_row)

# Create buttons
start_button_rect = pygame.Rect(BUTTON_LEFT_START, BUTTON_TOP, BUTTON_WIDTH, BUTTON_HEIGHT)
stop_button_rect = pygame.Rect(BUTTON_RIGHT_START, BUTTON_TOP, BUTTON_WIDTH, BUTTON_HEIGHT)

# Initialize variables
selected_text = ""
current_flash = None
flash_duration = 100  # Duration of flash in milliseconds
inter_flash_interval = 150  # Time between flashes in milliseconds
next_flash_time = 0
status_text = "Press 'Start Flashing' to begin"

# Serial Communication Setup
try:
    # Ensure the correct port where the Arduino is connected
    arduino_port = 'COM4'  # Change this to the correct port
    baud_rate = 115200
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)  # Open serial port
    serial_connected = True
except:
    # Handle case where serial connection fails
    serial_connected = False
    status_text = "Warning: Arduino not connected. Simulation mode only."

# To store the EEG signal values in a queue for processing
eeg_data_queue = collections.deque(maxlen=500)  # Store last 500 samples

# Function to draw rounded rectangle
def draw_rounded_rect(surface, color, rect, radius, border=0, border_color=None):
    """Draw a rounded rectangle"""
    rect = pygame.Rect(rect)
    
    # Draw the main rectangle
    pygame.draw.rect(surface, color, rect, border_radius=radius)
    
    # If border is specified, draw the border
    if border > 0:
        pygame.draw.rect(surface, border_color or color, rect, width=border, border_radius=radius)

# Functions to calculate normalized threshold (P300 detection)
def calculate_normalized_threshold(eeg_signal_window):
    """
    Calculate the normalized threshold for P300 detection:
    Z = (X - μ) / σ
    Where:
    - X: Current signal value
    - μ: Mean of the signal
    - σ: Standard deviation of the signal
    """
    mean = np.mean(eeg_signal_window)
    std_dev = np.std(eeg_signal_window)
    return (eeg_signal_window[-1] - mean) / std_dev if std_dev != 0 else 0

# Function to detect P300
def detect_p300(eeg_signal_window, threshold=2.7, window_size=0.5, sampling_rate=500):
    """
    Detects P300 based on normalized threshold.
    """
    normalized_threshold = calculate_normalized_threshold(eeg_signal_window)
    
    # If the normalized threshold exceeds the cutoff value, we detect a P300 event
    if normalized_threshold > threshold:
        return True
    return False

# Pygame setup for flashing and UI elements
def start_flashing():
    global flashing, status_text, next_flash_time
    flashing = True
    status_text = "Flashing active - detecting P300 signals..."
    next_flash_time = pygame.time.get_ticks()

def stop_flashing():
    global flashing, status_text
    flashing = False
    status_text = "Flashing stopped"
    reset_all_flashes()

def reset_all_flashes():
    for row in cells:
        for cell in row:
            cell["flash"] = False

def flash_row_or_column(is_row, index):
    reset_all_flashes()
    if is_row:
        for cell in cells[index]:
            cell["flash"] = True
    else:
        for row in cells:
            row[index]["flash"] = True

def handle_button_click(pos):
    global flashing
    
    if start_button_rect.collidepoint(pos) and not flashing:
        start_flashing()
    elif stop_button_rect.collidepoint(pos) and flashing:
        stop_flashing()

# Text rendering and UI drawing
def draw_screen():
    screen.fill(BACKGROUND)
    
    # Draw title
    title_text = font_title.render("P300 Speller Simulation", True, TITLE_COLOR)
    screen.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, int(HEIGHT * 0.03)))
    
    # Draw status
    status_surface = font_status.render(status_text, True, BLACK)
    screen.blit(status_surface, (STATUS_LEFT + STATUS_WIDTH // 2 - status_surface.get_width() // 2, STATUS_TOP))
    
    # Draw cells
    for row_idx, row in enumerate(cells):
        for col_idx, cell in enumerate(row):
            if cell["flash"]:
                color = YELLOW
            elif cell["selected"]:
                color = GREEN
            else:
                color = WHITE
            
            draw_rounded_rect(screen, color, cell["rect"], CELL_RADIUS, border=1, border_color=BORDER_COLOR)
            text = font_cell.render(cell["char"], True, BLACK)
            screen.blit(text, (cell["rect"].centerx - text.get_width() // 2, cell["rect"].centery - text.get_height() // 2))
    
    # Draw text box and selected text
    shadow_rect = pygame.Rect(TEXT_BOX_LEFT + SHADOW_OFFSET, TEXT_BOX_TOP + SHADOW_OFFSET, TEXT_BOX_WIDTH, TEXT_BOX_HEIGHT)
    draw_rounded_rect(screen, GRAY, shadow_rect, TEXT_BOX_RADIUS)
    draw_rounded_rect(screen, TEXT_BOX_BG, (TEXT_BOX_LEFT, TEXT_BOX_TOP, TEXT_BOX_WIDTH, TEXT_BOX_HEIGHT), TEXT_BOX_RADIUS, border=1, border_color=BORDER_COLOR)
    
    if selected_text:
        text_surface = font_text.render(selected_text, True, BLACK)
        screen.blit(text_surface, (TEXT_BOX_LEFT + 20, TEXT_BOX_TOP + 20))
    
    # Draw buttons
    start_color = BUTTON_COLOR if not flashing else BUTTON_DISABLED
    stop_color = BUTTON_DISABLED if not flashing else BUTTON_COLOR
    
    draw_rounded_rect(screen, start_color, start_button_rect, BUTTON_RADIUS)
    draw_rounded_rect(screen, stop_color, stop_button_rect, BUTTON_RADIUS)
    
    start_text = font_button.render("Start Flashing", True, WHITE)
    stop_text = font_button.render("Stop Flashing", True, WHITE)
    
    screen.blit(start_text, (start_button_rect.centerx - start_text.get_width() // 2, 
                            start_button_rect.centery - start_text.get_height() // 2))
    screen.blit(stop_text, (stop_button_rect.centerx - stop_text.get_width() // 2, 
                            stop_button_rect.centery - stop_text.get_height() // 2))
    
    pygame.display.flip()

# Main game loop
clock = pygame.time.Clock()
running = True
flashing = False

while running:
    current_time = pygame.time.get_ticks()
    
    # Read EEG data from Arduino if connected
    if serial_connected and ser.in_waiting > 0:
        try:
            eeg_value = int(ser.readline().decode().strip())  # Read and decode the EEG value
            eeg_data_queue.append(eeg_value)  # Add it to the queue

            # If we have enough data for a window
            if len(eeg_data_queue) == 500 and flashing:
                # Check for P300 detection using the EEG signal
                detected = detect_p300(list(eeg_data_queue))
                if detected and current_flash is not None:
                    is_row, index = current_flash
                    # If a row was flashed, we know the row of the character
                    if is_row:
                        row_idx = index
                        # Need to flash all columns to determine the column
                        # (Simplified for this example - in reality would need more flashing)
                    else:
                        col_idx = index
                        # Need to flash all rows to determine the row
                        # (Simplified for this example)
                    
                    # Simulate character selection (this is oversimplified)
                    # In an actual implementation, you'd need to track which row and column had P300 responses
                    # and only select a character after both are identified
                    selected_text += "A"  # Replace with actual character mapping
        except:
            pass  # Handle serial reading errors gracefully
    
    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            handle_button_click(event.pos)
    
    # Handle flashing logic
    if flashing and current_time >= next_flash_time:
        if current_flash is None:  # Start a new flash
            is_row = random.random() < 0.5
            num_elements = len(matrix) if is_row else len(matrix[0])
            index = random.randint(0, num_elements - 1)
            
            current_flash = (is_row, index)
            flash_row_or_column(is_row, index)
            next_flash_time = current_time + flash_duration
        else:  # End current flash
            reset_all_flashes()
            current_flash = None
            
            next_flash_time = current_time + inter_flash_interval
    
    # Draw everything
    draw_screen()
    
    # Cap the frame rate
    clock.tick(60)

# Clean up
if serial_connected:
    ser.close()
pygame.quit()
sys.exit()