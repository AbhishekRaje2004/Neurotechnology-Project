# P300 Brain-Computer Interface Project

This repository contains a complete implementation of a P300-based Brain-Computer Interface (BCI) system for text entry. The system utilizes the P300 event-related potential, a positive deflection in EEG approximately 300ms after a rare, task-relevant stimulus, to enable text input.

## Project Overview

This BCI system combines advanced signal processing techniques with machine learning algorithms to achieve robust classification of P300 responses. Key components include:

1. **Data Acquisition**: Arduino-based EEG acquisition from 8 electrodes
2. **Signal Processing**: Real-time preprocessing, artifact rejection, and feature extraction
3. **P300 Classification**: Ensemble learning approach combining multiple classifiers
4. **Text Entry Interface**: Row-column P300 speller paradigm
5. **Analysis & Visualization**: Tools for data analysis and performance evaluation

## Repository Structure

- **`main.py`**: Main application entry point
- **`model_statistics.py`**: GUI tool for visualizing and analyzing P300 data
- **`acquisition_arduino.py`**: Interface for Arduino-based EEG acquisition
- **`preprocessing.py`**: Signal preprocessing functions
- **`epoching.py`**: Event-related potential extraction
- **`p300_classifier.py`**: P300 classification algorithms
- **`speller.py`**: P300 speller interface implementation
- **`stimulus.py`**: Stimulus presentation for P300 paradigms
- **`visualization.py`**: Visualization tools and data plotting
- **`analysis.py`**: Data analysis functions
- **`data/`**: Directory containing model files and session data
- **`P300_BCI_Paper.txt`**: Technical paper describing the project (IEEE format)

## Getting Started

1. **Hardware Requirements**:
   - Arduino-based EEG acquisition system
   - Electrodes for EEG recording
   - Computer with Python 3.8+ installed

2. **Software Setup**:
   ```bash
   pip install numpy scipy matplotlib psychopy scikit-learn tkinter pickle
   ```

3. **Running the Application**:
   ```bash
   python main.py
   ```

4. **Data Visualization**:
   ```bash
   python model_statistics.py
   ```

## Key Features

- Real-time P300 detection with ~87% accuracy
- Information transfer rate of ~25 bits/minute
- Interactive visualization of ERPs and frequency components
- Comprehensive model statistics and performance metrics
- Support for multiple experimental paradigms (oddball, P300 speller)

## Usage Examples

### Calibration Mode
```bash
python main.py --mode calibration
```

### Online Speller Mode
```bash
python main.py --mode speller
```

### Data Analysis
```bash
python model_statistics.py
```

## Performance Metrics

- Classification accuracy: 87.2%
- Area under ROC curve: 0.91
- Sensitivity: 83.5%
- Specificity: 91.3%
- Character accuracy: 83.7%
- Information transfer rate: 25.4 bits/minute

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

This work was developed under the guidance of Dr. Kousik Sridharan Sarthy. Special thanks for the mentorship and expertise in BCI research and development.
