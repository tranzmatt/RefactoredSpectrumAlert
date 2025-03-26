#!/usr/bin/env python3
"""
SpectrumAlert MiniTrainer - Lightweight ML model trainer for RF fingerprinting and anomaly detection.
"""

import argparse
import os
import sys
from collections import Counter
from typing import List, Optional, Tuple, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

# Import common utilities
from spectrum_utils import (
    CSVDataHandler, safe_float_conversion
)

# Default constants
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CONTAMINATION = 0.05
DEFAULT_DEVICE_COUNT = 5  # Number of devices to simulate


class DataHandler:
    """Handles data loading and preparation."""
    
    @staticmethod
    def load_from_csv(filename: str) -> np.ndarray:
        """
        Load feature data from a CSV file.
        
        Args:
            filename: Path to the CSV file
            
        Returns:
            NumPy array of feature data
            
        Raises:
            FileNotFoundError: If the file doesn't exist
            ValueError: If the file is empty or has invalid format
        """
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")
            
        # Use the common CSVDataHandler for reading
        csv_handler = CSVDataHandler(filename)
        header, rows = csv_handler.read_data()
        
        if not rows:
            raise ValueError("No data found in the CSV file.")
        
        features = []
        for row in rows:
            if len(row) < 2:
                print(f"Warning: Skipping invalid row: {row}")
                continue
                
            try:
                # First column is frequency, rest are features
                feature_values = [safe_float_conversion(value) for value in row[1:]]
                features.append(feature_values)
            except Exception as e:
                print(f"Warning: Error parsing row {row}: {e}")
        
        if not features:
            raise ValueError("No valid data found in the CSV file.")
            
        return np.array(features)
    
    @staticmethod
    def generate_labels(data_length: int, device_count: int) -> List[str]:
        """
        Generate simulated device labels for training.
        
        Args:
            data_length: Number of samples
            device_count: Number of unique devices to simulate
            
        Returns:
            List of device label strings
        """
        return [f"Device_{i % device_count}" for i in range(data_length)]


class LiteModelTrainer:
    """Handles training of lightweight ML models."""
    
    def __init__(self, device_count: int = DEFAULT_DEVICE_COUNT, 
                 test_size: float = DEFAULT_TEST_SIZE, 
                 random_state: int = DEFAULT_RANDOM_STATE,
                 contamination: float = DEFAULT_CONTAMINATION):
        """
        Initialize the model trainer.
        
        Args:
            device_count: Number of devices to simulate
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility
            contamination: Contamination parameter for IsolationForest
        """
        self.device_count = device_count
        self.test_size = test_size
        self.random_state = random_state
        self.contamination = contamination
        self.data_handler = DataHandler()
        
    def train_models(self, features: np.ndarray) -> Tuple[Optional[RandomForestClassifier], Optional[IsolationForest]]:
        """
        Train RF fingerprinting and anomaly detection models.
        
        Args:
            features: Feature array
            
        Returns:
            Tuple of (fingerprinting_model, anomaly_model) or (None, None) if training fails
        """
        # Ensure sufficient data for training
        if len(features) < 2:
            print("Not enough data to train the model.")
            return None, None
            
        try:
            # Generate labels for the entire dataset
            labels = self.data_handler.generate_labels(len(features), self.device_count)
            
            # Train fingerprinting model
            fingerprint_model = self._train_fingerprinting_model(features, labels)
            
            # Train anomaly detection model
            anomaly_model = self._train_anomaly_model(features)
            
            return fingerprint_model, anomaly_model
            
        except Exception as e:
            print(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
            return None, None
    
    def _train_fingerprinting_model(self, features: np.ndarray, labels: List[str]) -> RandomForestClassifier:
        """
        Train a lightweight Random Forest model for RF fingerprinting.
        
        Args:
            features: Feature array
            labels: Device labels
            
        Returns:
            Trained RandomForestClassifier
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, 
            labels,
            test_size=self.test_size, 
            random_state=self.random_state
        )
        
        # Determine cross-validation splits based on class distribution
        class_counts = Counter(y_train)
        min_samples_per_class = min(class_counts.values())
        max_cv_splits = min(3, min_samples_per_class)  # Simplified cross-validation
        
        print(f"Using {max_cv_splits}-fold cross-validation (lite version).")
        
        # Create lightweight RandomForest
        model = RandomForestClassifier(
            n_estimators=50,  # Reduced number of trees for faster training
            max_depth=10,     # Reduced depth for lower complexity
            random_state=self.random_state
        )
        
        # Train the model
        print("Training the RF fingerprinting model (lite version)...")
        model.fit(X_train, y_train)
        
        # Evaluate the model
        self._evaluate_model(model, X_test, y_test, features, labels, max_cv_splits)
        
        return model
    
    def _train_anomaly_model(self, features: np.ndarray) -> IsolationForest:
        """
        Train a lightweight Isolation Forest model for anomaly detection.
        
        Args:
            features: Feature array
            
        Returns:
            Trained IsolationForest
        """
        print("Training the IsolationForest model for anomaly detection (lite version)...")
        
        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=50  # Reduced for lite version
        )
        
        model.fit(features)
        print("Anomaly detection model trained successfully.")
        
        return model
    
    def _evaluate_model(self, model: RandomForestClassifier, X_test: np.ndarray, 
                        y_test: List[str], features: np.ndarray, labels: List[str], 
                        cv_splits: int) -> None:
        """
        Evaluate model performance with multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test feature data
            y_test: Test labels
            features: All features for cross-validation
            labels: All labels for cross-validation
            cv_splits: Number of cross-validation splits
        """
        # Test set evaluation
        y_pred = model.predict(X_test)
        print(f"Classification accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        
        # Detailed classification report
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        # Cross-validation
        skf = StratifiedKFold(n_splits=cv_splits)
        
        try:
            cv_scores = cross_val_score(model, features, labels, cv=skf)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean cross-validation score: {np.mean(cv_scores) * 100:.2f}%")
        except Exception as e:
            print(f"Warning: Error during cross-validation: {e}")


class ModelSaver:
    """Handles saving trained models to disk."""
    
    @staticmethod
    def save_model(model: Any, filename: str) -> bool:
        """
        Save a trained model to a file.
        
        Args:
            model: Model to save
            filename: Path to save the model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create directory if it doesn't exist
            directory = os.path.dirname(filename)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                
            joblib.dump(model, filename)
            print(f"Model saved to {filename}")
            return True
            
        except Exception as e:
            print(f"Error saving model to {filename}: {e}")
            return False


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(
        description="Lightweight ML trainer for RF fingerprinting and anomaly detection."
    )
    
    parser.add_argument("-i", "--input", type=str, default="collected_data_lite.csv",
                        help="Path to the collected lite data file (default: collected_data_lite.csv)")
    parser.add_argument("-a", "--anomaly", type=str, default="anomaly_detection_model_lite.pkl",
                        help="Path to the output lite anomaly file (default: anomaly_detection_model_lite.pkl)")
    parser.add_argument("-f", "--fingerprint", type=str, default="rf_fingerprinting_model_lite.pkl",
                        help="Path to the output lite fingerprint file (default: rf_fingerprinting_model_lite.pkl)")
    parser.add_argument("-d", "--devices", type=int, default=DEFAULT_DEVICE_COUNT,
                        help=f"Number of devices to simulate (default: {DEFAULT_DEVICE_COUNT})")
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Display configuration
        print(f"Loading data from {args.input}...")
        print(f"Fingerprint model will be saved to {args.fingerprint}")
        print(f"Anomaly model will be saved to {args.anomaly}")
        print(f"Simulating {args.devices} devices")
        
        # Load data
        try:
            data_handler = DataHandler()
            features = data_handler.load_from_csv(args.input)
            print(f"Loaded {len(features)} samples with {features.shape[1]} features each")
            print(f"Sample features (first 5):\n{features[:5]}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1
            
        # Train models
        trainer = LiteModelTrainer(device_count=args.devices)
        fingerprint_model, anomaly_model = trainer.train_models(features)
        
        print(f"Saving to {args.fingerprint} / {args.anomaly}...")
        
        # Save models
        model_saver = ModelSaver()
        
        if fingerprint_model is not None:
            model_saver.save_model(fingerprint_model, args.fingerprint)
        
        if anomaly_model is not None:
            model_saver.save_model(anomaly_model, args.anomaly)
            
        print("Model training and saving completed successfully.")
        return 0
        
    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
