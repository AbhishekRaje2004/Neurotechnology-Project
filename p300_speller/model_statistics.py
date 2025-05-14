import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import signal
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score

# Acknowledgements:
# This work was developed under the guidance of Dr. Kousik Sridharan Sarthy.
# Special thanks for the mentorship and expertise in BCI research and development.

class PKLViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("P300 Model Statistics Viewer")
        self.root.geometry("1100x900")
        self.data = None
        self.filename = None
        
        self.setup_ui()
        
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Controls frame
        controls_frame = ttk.Frame(main_frame)
        controls_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)
        
        # File selection
        ttk.Label(controls_frame, text="Select PKL File:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(controls_frame, textvariable=self.file_var, width=40)
        self.file_combo.pack(side=tk.LEFT, padx=(0, 10))
        self.file_combo.bind("<<ComboboxSelected>>", self.on_file_selected)
        
        ttk.Button(controls_frame, text="Browse...", command=self.browse_file).pack(side=tk.LEFT)
        ttk.Button(controls_frame, text="Load", command=self.load_selected_file).pack(side=tk.LEFT, padx=(10, 0))
        ttk.Button(controls_frame, text="Refresh Files", command=self.update_file_list).pack(side=tk.LEFT, padx=(10, 0))
        
        # Tabs for different views
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Summary tab
        self.summary_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.summary_frame, text="Summary")
        
        # Statistics tab
        self.stats_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.stats_frame, text="Statistics")
        
        # ERP Plot tab
        self.erp_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.erp_frame, text="ERP Plots")
        
        # Frequency tab
        self.freq_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.freq_frame, text="Frequency Analysis")
        
        # Raw data tab
        self.raw_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.raw_frame, text="Raw Data")
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready. Select a PKL file to view statistics.")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Initialize file list
        self.update_file_list()
          def update_file_list(self):
        """Update the list of available pkl files"""
        pkl_files = []
        current_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(current_dir, "data")
        
        if os.path.exists(data_dir):
            for file in os.listdir(data_dir):
                if file.endswith(".pkl"):
                    pkl_files.append(file)
        
        self.file_combo['values'] = pkl_files
        
        if pkl_files and not self.file_combo.get():
            self.file_combo.current(0)
            
        self.status_var.set(f"Found {len(pkl_files)} PKL files. Select one to view statistics.")
    
    def browse_file(self):
        """Open file dialog to select PKL file"""
        filename = filedialog.askopenfilename(
            title="Select PKL File",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            # Extract just the filename for display
            basename = os.path.basename(filename)
            
            # Update combobox
            values = list(self.file_combo['values'])
            if basename not in values:
                values.append(basename)
                self.file_combo['values'] = values
                
            self.file_var.set(basename)
            self.filename = filename
            self.load_selected_file()
      def on_file_selected(self, event):
        """Handle file selection from combobox"""
        selected = self.file_var.get()
        if selected:
            self.filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", selected)
            self.status_var.set(f"Selected file: {selected}")
    
    def load_selected_file(self):
        """Load the selected PKL file"""
        if not self.filename:
            self.status_var.set("No file selected")
            return
        
        try:
            with open(self.filename, 'rb') as f:
                self.data = pickle.load(f)
                
            self.status_var.set(f"Loaded {os.path.basename(self.filename)} successfully")
            
            # Update all tabs
            self.update_summary_tab()
            self.update_stats_tab()
            self.update_erp_tab()
            self.update_freq_tab()
            self.update_raw_tab()
            
        except Exception as e:
            self.status_var.set(f"Error loading file: {str(e)}")
    
    def clear_frame(self, frame):
        """Clear all widgets from a frame"""
        for widget in frame.winfo_children():
            widget.destroy()
    
    def update_summary_tab(self):
        """Update the summary tab with basic information"""
        self.clear_frame(self.summary_frame)
        
        if not self.data:
            ttk.Label(self.summary_frame, text="No data loaded").pack(pady=20)
            return
        
        # Create scrollable text widget
        text_frame = ttk.Frame(self.summary_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # File information
        text_widget.insert(tk.END, f"File: {os.path.basename(self.filename)}\n", "heading")
        text_widget.insert(tk.END, f"Path: {self.filename}\n\n")
        
        # Basic data structure
        text_widget.insert(tk.END, "Data Structure:\n", "heading")
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                shape_str = f"array shape {value.shape}, dtype {value.dtype}"
                text_widget.insert(tk.END, f"  • {key}: {shape_str}\n")
            else:
                text_widget.insert(tk.END, f"  • {key}: {type(value).__name__}\n")
        
        text_widget.insert(tk.END, "\n")
        
        # More specific information based on data type
        if 'epochs' in self.data and 'labels' in self.data:
            # Session data
            epochs = self.data['epochs']
            labels = self.data['labels']
            
            text_widget.insert(tk.END, "Session Information:\n", "heading")
            text_widget.insert(tk.END, f"  • Total epochs: {len(epochs)}\n")
            text_widget.insert(tk.END, f"  • Target epochs: {np.sum(labels == 1)}\n")
            text_widget.insert(tk.END, f"  • Non-target epochs: {np.sum(labels == 0)}\n")
            text_widget.insert(tk.END, f"  • Sampling frequency: {self.data.get('fs', 'unknown')} Hz\n")
            
            if 'timestamp' in self.data:
                text_widget.insert(tk.END, f"  • Session timestamp: {self.data['timestamp']}\n")
        
        elif 'pipeline' in self.data:
            # Classifier model
            text_widget.insert(tk.END, "Model Information:\n", "heading")
            text_widget.insert(tk.END, f"  • Model type: P300 Classifier\n")
            
            if 'cv_scores' in self.data:
                cv_scores = self.data['cv_scores']
                text_widget.insert(tk.END, f"  • Cross-validation scores: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}\n")
            
        elif 'eeg_data' in self.data and 'timestamps' in self.data:
            # Raw EEG data
            text_widget.insert(tk.END, "Raw Data Information:\n", "heading")
            text_widget.insert(tk.END, f"  • Samples: {len(self.data['eeg_data'])}\n")
            text_widget.insert(tk.END, f"  • Duration: {self.data['timestamps'][-1] - self.data['timestamps'][0]:.2f} seconds\n")
            
            if 'markers' in self.data:
                markers = self.data['markers']
                text_widget.insert(tk.END, f"  • Event markers: {len(markers)}\n")
        
        # Tag configuration
        text_widget.tag_config("heading", font=("Arial", 10, "bold"))
    
    def update_stats_tab(self):
        """Update the statistics tab with model performance metrics"""
        self.clear_frame(self.stats_frame)
        
        if not self.data:
            ttk.Label(self.stats_frame, text="No data loaded").pack(pady=20)
            return
        
        # Check if this is a model file or has enough data to compute statistics
        if 'epochs' not in self.data or 'labels' not in self.data:
            ttk.Label(self.stats_frame, text="This file doesn't contain classifier performance data").pack(pady=20)
            return
            
        # Create scrollable frame for statistics
        outer_frame = ttk.Frame(self.stats_frame)
        outer_frame.pack(fill=tk.BOTH, expand=True)
        
        # Add scrollbar
        v_scrollbar = ttk.Scrollbar(outer_frame, orient="vertical")
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Create canvas for scrolling
        canvas = tk.Canvas(outer_frame, yscrollcommand=v_scrollbar.set)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        v_scrollbar.config(command=canvas.yview)
        
        # Create frame inside canvas for content
        content_frame = ttk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=content_frame, anchor="nw")
        
        # Configure the canvas to resize with the window and update scrollregion
        def configure_canvas(event):
            canvas.configure(scrollregion=canvas.bbox("all"))
            canvas.itemconfig(canvas_window, width=event.width)
        
        canvas.bind("<Configure>", configure_canvas)
        content_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        
        # Enable mouse wheel scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        try:
            # Stats container
            stats_frame = ttk.LabelFrame(content_frame, text="Classification Statistics")
            stats_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
            
            # Basic stats with validation
            epochs = self.data['epochs']
            labels = self.data['labels']
            
            # Ensure arrays have consistent lengths
            if isinstance(labels, np.ndarray) and isinstance(epochs, np.ndarray):
                if len(labels) > len(epochs):
                    labels = labels[:len(epochs)]
                elif len(epochs) > len(labels):
                    epochs = epochs[:len(labels)]
            
            # Check if we have model prediction data
            if 'predictions' in self.data:
                y_true = self.data['labels']
                y_pred = self.data['predictions']
                
                # Confusion Matrix
                cm = confusion_matrix(y_true, y_pred)
                tn, fp, fn, tp = cm.ravel()
                
                fig_cm = plt.figure(figsize=(5, 4))
                plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
                plt.title('Confusion Matrix')
                plt.colorbar()
                
                classes = ['Non-target', 'Target']
                tick_marks = np.arange(len(classes))
                plt.xticks(tick_marks, classes, rotation=45)
                plt.yticks(tick_marks, classes)
                
                # Add text annotations
                thresh = cm.max() / 2.
                for i in range(cm.shape[0]):
                    for j in range(cm.shape[1]):
                        plt.text(j, i, format(cm[i, j], 'd'),
                                 horizontalalignment="center",
                                 color="white" if cm[i, j] > thresh else "black")
                
                plt.tight_layout()
                plt.ylabel('True label')
                plt.xlabel('Predicted label')
                
                cm_canvas = FigureCanvasTkAgg(fig_cm, master=stats_frame)
                cm_canvas.draw()
                cm_canvas.get_tk_widget().grid(row=1, column=0, sticky=tk.NSEW)
                
                # Performance metrics
                metrics_frame = ttk.LabelFrame(stats_frame, text="Performance Metrics")
                metrics_frame.grid(row=2, column=0, sticky=tk.EW, pady=10)
                
                # Calculate metrics
                accuracy = (tp + tn) / (tp + tn + fp + fn)
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                
                # Display metrics
                metrics = [
                    ("Accuracy", accuracy),
                    ("Sensitivity (Recall)", sensitivity),
                    ("Specificity", specificity),
                    ("Precision", precision),
                    ("F1 Score", f1),
                ]
                
                for i, (name, value) in enumerate(metrics):
                    ttk.Label(metrics_frame, text=f"{name}:").grid(row=i, column=0, sticky=tk.W, padx=5, pady=2)
                    ttk.Label(metrics_frame, text=f"{value:.4f}").grid(row=i, column=1, sticky=tk.E, padx=5, pady=2)
            
            else:
                # Basic statistics about the epochs
                if len(epochs) > 0:
                    # Info frame to display basic statistics
                    info_frame = ttk.Frame(stats_frame)
                    info_frame.pack(fill=tk.X, padx=10, pady=10)
                    
                    # Display file info and dimensions
                    ttk.Label(info_frame, text=f"File: {os.path.basename(self.filename)}", font=("Arial", 10, "bold")).grid(
                        row=0, column=0, columnspan=2, sticky=tk.W, pady=(0,5))
                    
                    ttk.Label(info_frame, text=f"Epochs shape: {epochs.shape}").grid(
                        row=1, column=0, sticky=tk.W)
                    ttk.Label(info_frame, text=f"Labels shape: {labels.shape if isinstance(labels, np.ndarray) else len(labels)}").grid(
                        row=2, column=0, sticky=tk.W)
                    
                    # Calculate stats about target vs non-target
                    try:
                        # Convert labels to numpy array if needed
                        if not isinstance(labels, np.ndarray):
                            labels = np.array(labels)
                        
                        # Find target and non-target indices
                        target_indices = np.where(labels == 1)[0]
                        non_target_indices = np.where(labels == 0)[0]
                        
                        # Only use valid indices (within bounds of epochs array)
                        target_indices = target_indices[target_indices < len(epochs)]
                        non_target_indices = non_target_indices[non_target_indices < len(epochs)]
                        
                        # Create a sub-frame for epoch statistics
                        metrics_frame = ttk.LabelFrame(stats_frame, text="Epoch Statistics")
                        metrics_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
                        
                        # Display counts
                        ttk.Label(metrics_frame, text="Dataset Statistics:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5,5))
                        ttk.Label(metrics_frame, text=f"Total epochs: {len(epochs)}").pack(anchor=tk.W, padx=10)
                        ttk.Label(metrics_frame, text=f"Target epochs: {len(target_indices)}").pack(anchor=tk.W, padx=10)
                        ttk.Label(metrics_frame, text=f"Non-target epochs: {len(non_target_indices)}").pack(anchor=tk.W, padx=10)
                        
                        if len(target_indices) > 0 and len(non_target_indices) > 0:
                            ratio = len(target_indices) / len(non_target_indices)
                            ttk.Label(metrics_frame, text=f"Target/Non-target ratio: {ratio:.3f}").pack(anchor=tk.W, padx=10)
                            
                            # P300 metrics section
                            ttk.Separator(metrics_frame, orient='horizontal').pack(fill=tk.X, pady=10)
                            ttk.Label(metrics_frame, text="P300 Response Analysis:", font=("Arial", 10, "bold")).pack(anchor=tk.W, pady=(5,5))
                            
                            # Handle epoch dimensions correctly
                            channel_idx = 0  # Default to first channel
                            
                            # Determine the shape of epochs and handle accordingly
                            if len(epochs.shape) == 3:  # (trials, channels, samples)
                                n_epochs, n_channels, n_samples = epochs.shape
                            elif len(epochs.shape) == 2:  # (trials, samples)
                                n_epochs, n_samples = epochs.shape
                                n_channels = 1
                                # Reshape to expected 3D format
                                epochs = epochs.reshape(n_epochs, 1, n_samples)
                            else:
                                ttk.Label(metrics_frame, text=f"Unsupported epoch shape: {epochs.shape}").pack(pady=10)
                                return
                            
                            # Calculate timing info
                            fs = self.data.get('fs', 250)  # Default sample rate
                            pre = self.data.get('pre', 0.2)  # Pre-stimulus time
                            post = self.data.get('post', 0.8)  # Post-stimulus time
                            
                            # Calculate P300 window (typically 250-500ms post-stimulus)
                            # First convert to samples safely
                            total_duration = pre + post
                            samples_per_second = n_samples / total_duration
                            
                            # P300 window in seconds from stimulus onset
                            p300_start_sec = 0.25
                            p300_end_sec = 0.5
                            
                            # Convert to samples, ensuring we stay within bounds
                            p300_start_sample = int(min((pre + p300_start_sec) / total_duration * n_samples, n_samples-1))
                            p300_end_sample = int(min((pre + p300_end_sec) / total_duration * n_samples, n_samples))
                            
                            # Safety check for indices
                            if p300_start_sample >= n_samples:
                                p300_start_sample = max(0, n_samples // 2 - 10)
                            if p300_end_sample > n_samples:
                                p300_end_sample = n_samples
                            if p300_end_sample <= p300_start_sample:
                                p300_end_sample = min(n_samples, p300_start_sample + 5)
                                    
                            # Display the calculated window
                            ttk.Label(metrics_frame, text=f"P300 analysis window: {p300_start_sample}-{p300_end_sample} samples").pack(anchor=tk.W, padx=10)
                            ttk.Label(metrics_frame, text=f"Corresponds to approximately {p300_start_sec*1000:.0f}-{p300_end_sec*1000:.0f}ms post-stimulus").pack(anchor=tk.W, padx=10)
                            
                            try:
                                # Compute average waveforms
                                target_avg = np.mean(epochs[target_indices], axis=0)
                                non_target_avg = np.mean(epochs[non_target_indices], axis=0)
                                
                                # Extract P300 window
                                p300_window = slice(p300_start_sample, p300_end_sample)
                                
                                # Get peak amplitudes from averaged waveforms
                                target_peak = np.max(target_avg[channel_idx, p300_window])
                                non_target_peak = np.max(non_target_avg[channel_idx, p300_window])
                                
                                # Calculate metrics from individual epochs
                                target_peaks = np.max(epochs[target_indices, channel_idx, p300_window], axis=1)
                                non_target_peaks = np.max(epochs[non_target_indices, channel_idx, p300_window], axis=1)
                                
                                # Calculate statistics
                                target_mean = np.mean(target_peaks)
                                target_std = np.std(target_peaks)
                                non_target_mean = np.mean(non_target_peaks)
                                non_target_std = np.std(non_target_peaks)
                                
                                # P300 metrics frame
                                metrics_inner_frame = ttk.Frame(metrics_frame)
                                metrics_inner_frame.pack(fill=tk.X, padx=10, pady=5)
                                
                                # Display amplitude metrics
                                row = 0
                                ttk.Label(metrics_inner_frame, text="Target P300 amplitude:").grid(row=row, column=0, sticky=tk.W, padx=5)
                                ttk.Label(metrics_inner_frame, text=f"{target_mean:.2f} ± {target_std:.2f} µV").grid(row=row, column=1, sticky=tk.W, padx=5)
                                row += 1
                                
                                ttk.Label(metrics_inner_frame, text="Non-target amplitude:").grid(row=row, column=0, sticky=tk.W, padx=5)
                                ttk.Label(metrics_inner_frame, text=f"{non_target_mean:.2f} ± {non_target_std:.2f} µV").grid(row=row, column=1, sticky=tk.W, padx=5)
                                row += 1
                                
                                ttk.Label(metrics_inner_frame, text="P300 difference:").grid(row=row, column=0, sticky=tk.W, padx=5)
                                ttk.Label(metrics_inner_frame, text=f"{target_peak - non_target_peak:.2f} µV").grid(row=row, column=1, sticky=tk.W, padx=5)
                                row += 1
                                
                                # Add advanced metrics if possible
                                if target_std > 0:
                                    snr = target_mean / target_std
                                    ttk.Label(metrics_inner_frame, text="Target SNR:").grid(row=row, column=0, sticky=tk.W, padx=5)
                                    ttk.Label(metrics_inner_frame, text=f"{snr:.2f}").grid(row=row, column=1, sticky=tk.W, padx=5)
                                    row += 1
                                
                                if target_std > 0 or non_target_std > 0:
                                    pooled_std = np.sqrt((target_std**2 + non_target_std**2) / 2)
                                    if pooled_std > 0:
                                        effect_size = abs(target_mean - non_target_mean) / pooled_std
                                        ttk.Label(metrics_inner_frame, text="Effect size (Cohen's d):").grid(row=row, column=0, sticky=tk.W, padx=5)
                                        ttk.Label(metrics_inner_frame, text=f"{effect_size:.2f}").grid(row=row, column=1, sticky=tk.W, padx=5)
                                
                                # Add bar chart visualization with better sizing
                                fig = plt.figure(figsize=(9, 4), dpi=100)  # Increased width for better visualization
                                plt.bar(['Non-target', 'Target'], [non_target_mean, target_mean],
                                       yerr=[non_target_std, target_std], capsize=10,
                                       color=['red', 'blue'], alpha=0.7, width=0.6)  # Adjusted width
                                plt.title('P300 Amplitude Comparison')
                                plt.ylabel('Amplitude (µV)')
                                plt.grid(axis='y', linestyle='--', alpha=0.7)
                                
                                # Add y-axis limit with some padding
                                max_val = max(target_mean + target_std, non_target_mean + non_target_std)
                                min_val = min(target_mean - target_std, non_target_mean - non_target_std)
                                padding = 0.2 * (max_val - min_val)
                                plt.tight_layout()
                                
                                # Add the plot to the UI with specific size
                                chart_frame = ttk.Frame(stats_frame)
                                chart_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
                                
                                canvas = FigureCanvasTkAgg(fig, master=chart_frame)
                                canvas.draw()
                                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=10)
                                
                            except Exception as e:
                                ttk.Label(metrics_frame, text=f"Error calculating metrics: {str(e)}").pack(pady=10)
                                import traceback
                                traceback.print_exc()
                        else:
                            ttk.Label(metrics_frame, text="Not enough target or non-target epochs for analysis").pack(pady=20)
                    
                    except Exception as e:
                        ttk.Label(stats_frame, text=f"Error processing epochs: {str(e)}").pack(pady=20)
                        import traceback
                        traceback.print_exc()
                        
        except Exception as e:
            ttk.Label(content_frame, text=f"Error analyzing data: {str(e)}").pack(pady=20)
            import traceback
            traceback.print_exc()
            
        # Clean up mousewheel binding when tab is changed
        def _on_tab_change(event):
            canvas.unbind_all("<MouseWheel>")
        
        self.notebook.bind("<<NotebookTabChanged>>", _on_tab_change)
    
    def update_erp_tab(self):
        """Update the ERP tab with ERP plots"""
        self.clear_frame(self.erp_frame)
        
        if not self.data:
            ttk.Label(self.erp_frame, text="No data loaded").pack(pady=20)
            return
        
        # Check if this file has epoch data
        if 'epochs' not in self.data or 'labels' not in self.data:
            ttk.Label(self.erp_frame, text="This file doesn't contain ERP data").pack(pady=20)
            return
        
        # Get data
        epochs = self.data['epochs']
        labels = self.data['labels']
        fs = self.data.get('fs', 250)  # Default to 250Hz if not specified
        
        # Calculate time vector
        pre = self.data.get('pre', 0.2)
        post = self.data.get('post', 0.8)
        time_vector = np.linspace(-pre, post, epochs.shape[2])
        
        # Target vs non-target plots
        target_indices = np.where(labels == 1)[0]
        non_target_indices = np.where(labels == 0)[0]
        
        if len(target_indices) > 0 and len(non_target_indices) > 0:
            # Create ERP comparison plot
            fig = plt.figure(figsize=(10, 6))
            
            # Average ERPs
            target_avg = np.mean(epochs[target_indices], axis=0)
            non_target_avg = np.mean(epochs[non_target_indices], axis=0)
            
            plt.plot(time_vector, target_avg[0], 'b-', linewidth=2, label='Target (P300)')
            plt.plot(time_vector, non_target_avg[0], 'r-', linewidth=2, label='Non-target')
            
            # Add trigger line and P300 window
            plt.axvline(0, color='k', linestyle='--', label='Stimulus')
            plt.axvspan(0.25, 0.5, color='yellow', alpha=0.2, label='P300 window')
            
            plt.title('Event-Related Potentials: Target vs. Non-target')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (µV)')
            plt.grid(True)
            plt.legend()
            
            # Embed the figure in the tab
            canvas = FigureCanvasTkAgg(fig, master=self.erp_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add controls for filtering
            controls_frame = ttk.Frame(self.erp_frame)
            controls_frame.pack(fill=tk.X, padx=10, pady=(0, 10))
            
            ttk.Label(controls_frame, text="Filter:").pack(side=tk.LEFT, padx=5)
            
            # Filter settings and button
            filter_button = ttk.Button(controls_frame, text="Apply Filter", 
                                     command=lambda: self.filter_and_update_erp(time_vector, target_avg, non_target_avg, canvas, fig))
            filter_button.pack(side=tk.RIGHT, padx=5)
            
            # Filter frequency entries
            ttk.Label(controls_frame, text="Low cutoff (Hz):").pack(side=tk.LEFT, padx=(20, 5))
            self.low_cutoff_var = tk.StringVar(value="1")
            ttk.Entry(controls_frame, textvariable=self.low_cutoff_var, width=5).pack(side=tk.LEFT)
            
            ttk.Label(controls_frame, text="High cutoff (Hz):").pack(side=tk.LEFT, padx=(20, 5))
            self.high_cutoff_var = tk.StringVar(value="20")
            ttk.Entry(controls_frame, textvariable=self.high_cutoff_var, width=5).pack(side=tk.LEFT)
            
        else:
            ttk.Label(self.erp_frame, text="Not enough data to plot ERPs").pack(pady=20)
    
    def filter_and_update_erp(self, time_vector, target_avg, non_target_avg, canvas, fig):
        """Apply filter to ERP data and update plot"""
        try:
            # Get filter parameters
            low_cutoff = float(self.low_cutoff_var.get())
            high_cutoff = float(self.high_cutoff_var.get())
            
            # Apply filter
            fs = self.data.get('fs', 250)  # Default to 250Hz if not specified
            b, a = signal.butter(4, [low_cutoff, high_cutoff], fs=fs, btype='band')
            
            filtered_target = signal.filtfilt(b, a, target_avg[0])
            filtered_non_target = signal.filtfilt(b, a, non_target_avg[0])
            
            # Update plot
            plt.figure(fig.number)
            plt.clf()
            
            plt.plot(time_vector, filtered_target, 'b-', linewidth=2, label='Target (P300)')
            plt.plot(time_vector, filtered_non_target, 'r-', linewidth=2, label='Non-target')
            
            # Add trigger line and P300 window
            plt.axvline(0, color='k', linestyle='--', label='Stimulus')
            plt.axvspan(0.25, 0.5, color='yellow', alpha=0.2, label='P300 window')
            
            plt.title(f'Filtered ERPs: {low_cutoff}-{high_cutoff} Hz')
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude (µV)')
            plt.grid(True)
            plt.legend()
            
            canvas.draw()
            
            self.status_var.set(f"Applied {low_cutoff}-{high_cutoff} Hz filter to ERPs")
            
        except Exception as e:
            self.status_var.set(f"Error applying filter: {str(e)}")
    
    def update_freq_tab(self):
        """Update the frequency analysis tab"""
        self.clear_frame(self.freq_frame)
        
        if not self.data:
            ttk.Label(self.freq_frame, text="No data loaded").pack(pady=20)
            return
        
        # Check if this file has epoch data
        if 'epochs' not in self.data or 'labels' not in self.data:
            ttk.Label(self.freq_frame, text="This file doesn't contain frequency data").pack(pady=20)
            return
        
        # Get data
        epochs = self.data['epochs']
        labels = self.data['labels']
        fs = self.data.get('fs', 250)  # Default to 250Hz if not specified
        
        # Target vs non-target separation
        target_indices = np.where(labels == 1)[0]
        non_target_indices = np.where(labels == 0)[0]
        
        if len(target_indices) > 0 and len(non_target_indices) > 0:
            # Create frequency analysis plot
            fig = plt.figure(figsize=(10, 6))
            
            # Average ERPs
            target_avg = np.mean(epochs[target_indices], axis=0)[0]
            non_target_avg = np.mean(epochs[non_target_indices], axis=0)[0]
            
            # Compute power spectral density
            f_target, pxx_target = signal.welch(target_avg, fs=fs, nperseg=min(256, len(target_avg)))
            f_non_target, pxx_non_target = signal.welch(non_target_avg, fs=fs, nperseg=min(256, len(non_target_avg)))
            
            plt.semilogy(f_target, pxx_target, 'b', label='Target')
            plt.semilogy(f_non_target, pxx_non_target, 'r', label='Non-target')
            
            plt.title('Power Spectral Density')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power (µV²/Hz)')
            plt.grid(True)
            plt.legend()
            plt.xlim(0, 40)  # Limit to 0-40 Hz for better visualization
            
            # Embed the figure in the tab
            canvas = FigureCanvasTkAgg(fig, master=self.freq_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
            
            # Add additional frequency band analysis
            if 'epochs' in self.data:
                bands_frame = ttk.LabelFrame(self.freq_frame, text="Frequency Band Analysis")
                bands_frame.pack(fill=tk.X, padx=10, pady=10)
                
                # Calculate power in different frequency bands
                bands = [
                    ("Delta", 1, 4),
                    ("Theta", 4, 8),
                    ("Alpha", 8, 13),
                    ("Beta", 13, 30),
                    ("Gamma", 30, 40)
                ]
                
                for i, (band_name, low_freq, high_freq) in enumerate(bands):
                    # Calculate power in band
                    mask_target = (f_target >= low_freq) & (f_target <= high_freq)
                    mask_non_target = (f_non_target >= low_freq) & (f_non_target <= high_freq)
                    
                    power_target = np.mean(pxx_target[mask_target])
                    power_non_target = np.mean(pxx_non_target[mask_non_target])
                    
                    # Display band power
                    ttk.Label(bands_frame, text=f"{band_name} ({low_freq}-{high_freq} Hz):").grid(
                        row=i, column=0, sticky=tk.W, padx=5, pady=2)
                    ttk.Label(bands_frame, text=f"Target: {power_target:.2e} µV²/Hz").grid(
                        row=i, column=1, sticky=tk.E, padx=5, pady=2)
                    ttk.Label(bands_frame, text=f"Non-target: {power_non_target:.2e} µV²/Hz").grid(
                        row=i, column=2, sticky=tk.E, padx=5, pady=2)
                    ttk.Label(bands_frame, text=f"Ratio: {power_target/power_non_target:.2f}").grid(
                        row=i, column=3, sticky=tk.E, padx=5, pady=2)
        
        else:
            ttk.Label(self.freq_frame, text="Not enough data for frequency analysis").pack(pady=20)
    
    def update_raw_tab(self):
        """Update the raw data tab"""
        self.clear_frame(self.raw_frame)
        
        if not self.data:
            ttk.Label(self.raw_frame, text="No data loaded").pack(pady=20)
            return
        
        # Create a scrolled text area to show raw data
        text_frame = ttk.Frame(self.raw_frame)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        text_widget = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        text_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=text_widget.yview)
        
        # Insert data structure first
        text_widget.insert(tk.END, "Data Structure:\n\n", "heading")
        
        for key, value in self.data.items():
            if isinstance(value, np.ndarray):
                text_widget.insert(tk.END, f"{key}: {type(value).__name__} shape={value.shape}, dtype={value.dtype}\n")
                
                # Show some sample data for arrays
                if len(value.shape) == 1 and value.shape[0] > 0:
                    sample_data = value[:min(5, len(value))]
                    text_widget.insert(tk.END, f"  Sample data: {sample_data}\n")
                elif len(value.shape) > 1 and value.shape[0] > 0:
                    text_widget.insert(tk.END, f"  First element shape: {value[0].shape}\n")
            else:
                text_widget.insert(tk.END, f"{key}: {type(value).__name__} = {value}\n")
        
        text_widget.insert(tk.END, "\n")
        
        # Raw data statistics
        if 'eeg_data' in self.data and isinstance(self.data['eeg_data'], np.ndarray):
            eeg_data = self.data['eeg_data']
            text_widget.insert(tk.END, "EEG Data Statistics:\n\n", "heading")
            
            text_widget.insert(tk.END, f"Min value: {np.min(eeg_data)}\n")
            text_widget.insert(tk.END, f"Max value: {np.max(eeg_data)}\n")
            text_widget.insert(tk.END, f"Mean value: {np.mean(eeg_data)}\n")
            text_widget.insert(tk.END, f"Std. deviation: {np.std(eeg_data)}\n")
            
        # Tag configuration
        text_widget.tag_config("heading", font=("Arial", 10, "bold"))


def main():
    root = tk.Tk()
    app = PKLViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()