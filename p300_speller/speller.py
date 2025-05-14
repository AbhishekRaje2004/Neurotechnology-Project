import random
import time
import numpy as np
from psychopy import visual, core, event

class P300Speller:
    def __init__(self, win, ser, classifier=None):
        self.win = win
        self.ser = ser
        self.classifier = classifier
        
        # Standard 6x6 P300 speller matrix
        self.matrix = [
            ['A', 'B', 'C', 'D', 'E', 'F'],
            ['G', 'H', 'I', 'J', 'K', 'L'],
            ['M', 'N', 'O', 'P', 'Q', 'R'],
            ['S', 'T', 'U', 'V', 'W', 'X'],
            ['Y', 'Z', '1', '2', '3', '4'],
            ['5', '6', '7', '8', '9', '_']
        ]
        
        # Create visual elements for the matrix
        self.char_positions = {}
        self.char_stims = {}
        self.spacing = 0.15
        self.setup_matrix()
        
        # Text display - using black text for white background
        self.text_output = ""
        self.text_display = visual.TextStim(win, text="", pos=[0, -0.7], color='black', height=0.05)
        self.instruction = visual.TextStim(win, text="Focus on the character you want to type", 
                                          pos=[0, 0.7], color='black', height=0.05)
        
        # Flashing parameters
        self.flash_duration = 0.1
        self.isi = 0.1  # Inter-stimulus interval
        
    def setup_matrix(self):
        """Create the visual elements for the speller matrix"""
        y_pos = 0.35
        for row_idx, row in enumerate(self.matrix):
            x_pos = -0.35
            for col_idx, char in enumerate(row):
                pos = (x_pos, y_pos)
                self.char_positions[char] = (row_idx, col_idx)
                self.char_stims[char] = visual.TextStim(
                    self.win, text=char, pos=pos, 
                    color='black', height=0.05  # Black text for white background
                )
                x_pos += self.spacing
            y_pos -= self.spacing
    
    def flash_sequence(self, n_sequences=10, flash_mode='rc'):
        """Run a flash sequence (rows and columns or single items)
        
        Parameters:
        -----------
        n_sequences : int
            Number of complete sequences to run
        flash_mode : str
            'rc' for row/column flashing, 'single' for single item flashing
        
        Returns:
        --------
        epochs : list
            List of epochs data corresponding to each flash
        targets : list
            List of target/non-target labels for each flash
        flash_items : list
            List of flashed items (rows, columns, or characters)
        """
        if flash_mode == 'rc':
            flash_items = list(range(6)) + list(range(6, 12))  # 0-5: rows, 6-11: columns
        else:  # single mode
            flash_items = [(r, c) for r in range(6) for c in range(6)]
        
        all_flashes = []
        for _ in range(n_sequences):
            random.shuffle(flash_items)
            all_flashes.extend(flash_items)
        
        # Run the actual flashing sequence
        event_log = []
        for item in all_flashes:
            # Inter-stimulus interval
            self.win.flip()
            core.wait(self.isi)
            
            # Flash the item
            if flash_mode == 'rc':
                self.highlight_row_col(item)
            else:
                self.highlight_single(item)
                
            self.win.flip()
            
            # Send trigger
            trigger_code = item if isinstance(item, int) else item[0] * 6 + item[1]
            self.ser.write(f"{trigger_code}\n".encode())
            ts = time.time()
            event_log.append((ts, trigger_code))
            
            core.wait(self.flash_duration)
        
        return event_log
    
    def highlight_row_col(self, idx):
        """Highlight a row or column"""
        if idx < 6:  # Row
            for col in range(6):
                char = self.matrix[idx][col]
                self.char_stims[char].color = 'blue'  # Blue highlights for better contrast on white
                self.char_stims[char].draw()
        else:  # Column
            col = idx - 6
            for row in range(6):
                char = self.matrix[row][col]
                self.char_stims[char].color = 'blue'  # Blue highlights for better contrast on white
                self.char_stims[char].draw()
                
        # Draw the rest in black
        for r in range(6):
            for c in range(6):
                char = self.matrix[r][c]
                if (idx < 6 and r != idx) or (idx >= 6 and c != idx - 6):
                    self.char_stims[char].color = 'black'  # Black text for white background
                    self.char_stims[char].draw()
    
    def highlight_single(self, pos):
        """Highlight a single item at position (row, col)"""
        row, col = pos
        target_char = self.matrix[row][col]
        
        for r in range(6):
            for c in range(6):
                char = self.matrix[r][c]
                if r == row and c == col:
                    self.char_stims[char].color = 'blue'  # Blue highlights for better contrast on white
                else:
                    self.char_stims[char].color = 'black'  # Black text for white background
                self.char_stims[char].draw()
    
    def draw_matrix(self):
        """Draw the full matrix in black text on white background"""
        for char, stim in self.char_stims.items():
            stim.color = 'black'  # Black text for white background
            stim.draw()
        self.instruction.draw()
        self.text_display.draw()
        self.win.flip()
    
    def detect_p300(self, epochs, flash_events, target_pos):
        """Detect P300 response to identify selected item"""
        if self.classifier is None or not self.classifier.trained:
            raise ValueError("Classifier not available or not trained")
        
        # Get predictions for all epochs
        _, probabilities = self.classifier.predict(epochs)
        
        # Extract just the trigger codes from the flash_events
        trigger_codes = [event[1] for event in flash_events]
        
        # Determine if we're in row/column mode or single character mode
        is_rc_mode = all(isinstance(code, int) and code < 12 for code in trigger_codes)
        n_items = 12 if is_rc_mode else 36
        
        # Count how many flashes per item
        item_counts = {}
        for trigger in trigger_codes:
            if is_rc_mode:
                # For row/column mode, use the trigger directly
                item_idx = trigger
            else:
                # For single character mode, convert to a linear index
                # Assuming trigger code was calculated as row * 6 + col
                row = trigger // 6
                col = trigger % 6
                item_idx = row * 6 + col
            
            item_counts[item_idx] = item_counts.get(item_idx, 0) + 1
        
        # Calculate average score for each item
        item_scores = np.zeros(n_items)
        for i, trigger in enumerate(trigger_codes):
            if i >= len(probabilities):
                continue  # Skip if we don't have enough probability scores
            
            if is_rc_mode:
                item_idx = trigger
            else:
                row = trigger // 6
                col = trigger % 6
                item_idx = row * 6 + col
                
            item_scores[item_idx] += probabilities[i]
        
        # Divide by counts to get average
        for idx in range(n_items):
            if idx in item_counts and item_counts[idx] > 0:
                item_scores[idx] = item_scores[idx] / item_counts[idx]
        
        if is_rc_mode:  # Row/column mode
            # Find row and column with highest P300 score
            row_scores = item_scores[:6]
            col_scores = item_scores[6:]
            
            row = np.argmax(row_scores)
            col = np.argmax(col_scores)
            
            return self.matrix[row][col], (row_scores[row] + col_scores[col]) / 2
        else:  # Single character mode
            max_idx = np.argmax(item_scores)
            row, col = max_idx // 6, max_idx % 6
            return self.matrix[row][col], item_scores[max_idx]
    
    def run_calibration(self, target_char, n_sequences=10):
        """Run calibration sequence with known target"""
        target_pos = self.char_positions[target_char]
        
        # Highlight target briefly - using darker green for visibility on white background
        target_stim = self.char_stims[target_char]
        target_stim.color = 'darkgreen'
        target_stim.draw()
        self.win.flip()
        core.wait(1.0)
        
        # Draw the matrix
        self.draw_matrix()
        core.wait(1.0)
        
        # Run the flash sequence
        flash_events = self.flash_sequence(n_sequences=n_sequences)
        
        return flash_events, target_pos
    
    def add_character(self, char):
        """Add a character to the output text"""
        self.text_output += char
        self.text_display.text = self.text_output
    
    def run_typing_sequence(self, n_sequences=10):
        """Run a single character typing sequence"""
        # Draw the matrix
        self.draw_matrix()
        core.wait(1.0)
        
        # Run flash sequence
        flash_events = self.flash_sequence(n_sequences=n_sequences)
        
        return flash_events