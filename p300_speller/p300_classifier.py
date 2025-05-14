import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix
from preprocessing import preprocess_eeg

class P300Classifier:
    def __init__(self):
        self.pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('lda', LinearDiscriminantAnalysis(shrinkage='auto', solver='lsqr'))
        ])
        self.trained = False
        
    def extract_features(self, epochs):
        # Extract simple features: amplitude at expected P300 timepoints
        # P300 typically occurs 300-500ms post-stimulus
        fs = 250  # Sample rate
        p300_start = int(0.25 * fs)  # 250ms
        p300_end = int(0.5 * fs)     # 500ms
        
        # Extract mean amplitude in P300 window as feature
        features = np.mean(epochs[:, 0, p300_start:p300_end], axis=1).reshape(-1, 1)
        
        # Add additional features: peak amplitude and latency
        peak_amplitudes = np.max(epochs[:, 0, p300_start:p300_end], axis=1).reshape(-1, 1)
        peak_latencies = np.argmax(epochs[:, 0, p300_start:p300_end], axis=1).reshape(-1, 1)
        
        # Combine features
        features = np.hstack([features, peak_amplitudes, peak_latencies])
        
        return features
    
    def preprocess_epochs(self, epochs, fs=250):
        """Apply preprocessing to each epoch"""
        processed_epochs = np.zeros_like(epochs)
        
        for i in range(len(epochs)):
            # Apply preprocessing to each channel separately
            for ch in range(epochs.shape[1]):
                processed_epochs[i, ch] = preprocess_eeg(
                    epochs[i, ch],
                    fs=fs,
                    notch_freq=50,  # 50Hz for Europe, 60Hz for USA
                    bandpass=(1, 20),  # P300 is typically in 1-20Hz range
                    normalize=True
                )
                
        return processed_epochs
    
    def train(self, epochs, labels, preprocess=True):
        """Train the P300 classifier"""
        if preprocess:
            epochs = self.preprocess_epochs(epochs)
            
        features = self.extract_features(epochs)
        self.pipeline.fit(features, labels)
        self.trained = True
        
        # Training performance
        y_pred = self.pipeline.predict(features)
        accuracy = accuracy_score(labels, y_pred)
        cm = confusion_matrix(labels, y_pred)
        
        return accuracy, cm
    
    def predict(self, epochs, preprocess=True):
        """Make predictions on new data"""
        if not self.trained:
            raise ValueError("Classifier has not been trained yet")
        
        if preprocess:
            epochs = self.preprocess_epochs(epochs)
            
        features = self.extract_features(epochs)
        predictions = self.pipeline.predict(features)
        probabilities = self.pipeline.predict_proba(features)[:, 1]  # P(class=1)
        
        return predictions, probabilities
    
    def evaluate(self, epochs, labels, preprocess=True):
        """Evaluate classifier performance on test data"""
        if not self.trained:
            raise ValueError("Classifier has not been trained yet")
            
        if preprocess:
            epochs = self.preprocess_epochs(epochs)
            
        features = self.extract_features(epochs)
        predictions = self.pipeline.predict(features)
        accuracy = accuracy_score(labels, predictions)
        cm = confusion_matrix(labels, predictions)
        
        return accuracy, cm
    
    def save_model(self, filepath):
        """Save the trained model to disk"""
        if not self.trained:
            raise ValueError("Cannot save untrained model")
            
        model_data = {
            'pipeline': self.pipeline,
            'trained': self.trained
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
            
    def load_model(self, filepath):
        """Load a trained model from disk"""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file {filepath} not found")
            
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.pipeline = model_data['pipeline']
        self.trained = model_data['trained']
        
        print(f"Model loaded from {filepath}")
        return True