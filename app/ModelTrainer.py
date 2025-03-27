#!/usr/bin/env python3
"""
SpectrumAlert ModelTrainer - Train ML models for RF fingerprinting and anomaly detection.
"""

import argparse
import logging
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold

# Import common utilities
from spectrum_utils import (
    CSVDataHandler, safe_float_conversion
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelTrainer')


class ModelConfig:
    """Configuration for model training."""
    
    def __init__(self, input_file: str, fingerprint_model_file: str, anomaly_model_file: str,
                 test_size: float = 0.2, random_state: int = 42, contamination: float = 0.05,
                 n_jobs: int = -1, verbose: int = 1):
        """
        Initialize the model configuration.
        
        Args:
            input_file: Path to input data file
            fingerprint_model_file: Path to output fingerprinting model file
            anomaly_model_file: Path to output anomaly model file
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            contamination: Contamination parameter for IsolationForest
            n_jobs: Number of parallel jobs (-1 for all cores)
            verbose: Verbosity level
        """
        self.input_file = input_file
        self.fingerprint_model_file = fingerprint_model_file
        self.anomaly_model_file = anomaly_model_file
        self.test_size = test_size
        self.random_state = random_state
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.verbose = verbose


class DataLoader:
    """Handles loading and preparing training data."""
    
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
                logger.warning(f"Skipping invalid row: {row}")
                continue
                
            try:
                # First column is frequency, rest are features
                feature_values = [safe_float_conversion(value) for value in row[1:]]
                features.append(feature_values)
            except Exception as e:
                logger.warning(f"Error parsing row {row}: {e}")
        
        if not features:
            raise ValueError("No valid data found in the CSV file.")
            
        return np.array(features)
    
    @staticmethod
    def split_data(features: np.ndarray, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            features: Feature array
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (training_data, testing_data)
            
        Raises:
            ValueError: If there's not enough data to split
        """
        if len(features) < 2:
            raise ValueError("Not enough data to split into training and testing sets.")
            
        return train_test_split(features, test_size=test_size, random_state=random_state)
    
    @staticmethod
    def generate_simulated_labels(data_length: int, num_devices: int = 10) -> List[str]:
        """
        Generate simulated device labels for training.
        
        Args:
            data_length: Number of samples
            num_devices: Number of unique devices to simulate
            
        Returns:
            List of device label strings
        """
        return [f"Device_{i % num_devices}" for i in range(data_length)]


class ModelTrainer:
    """Handles training and evaluation of ML models."""
    
    def __init__(self, config: ModelConfig):
        """
        Initialize the model trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        self.data_loader = DataLoader()
        
    def train_fingerprinting_model(self, features: np.ndarray) -> Optional[RandomForestClassifier]:
        """
        Train a Random Forest model for device fingerprinting.
        
        Args:
            features: Feature array
            
        Returns:
            Trained RandomForestClassifier or None if training fails
        """
        try:
            # Split data
            X_train, X_test = self.data_loader.split_data(
                features, 
                self.config.test_size, 
                self.config.random_state
            )
            
            # Generate simulated labels
            train_labels = self.data_loader.generate_simulated_labels(len(X_train))
            test_labels = self.data_loader.generate_simulated_labels(len(X_test))
            
            # Configure cross-validation based on class distribution
            class_counts = Counter(train_labels)
            min_samples_per_class = min(class_counts.values())
            max_cv_splits = min(5, min_samples_per_class) 
            max_cv_splits = max(2, max_cv_splits)  # Ensure at least 2 splits
            
            logger.info(f"Using {max_cv_splits}-fold cross-validation (based on smallest class size).")
            
            # Set up hyperparameter tuning
            model = RandomForestClassifier(random_state=self.config.random_state)
            skf = StratifiedKFold(n_splits=max_cv_splits)
            
            param_grid = self._get_param_grid()
            
            grid_search = GridSearchCV(
                estimator=model, 
                param_grid=param_grid, 
                cv=skf, 
                n_jobs=self.config.n_jobs, 
                verbose=self.config.verbose
            )
            
            # Train with hyperparameter tuning
            logger.info("Training Random Forest model with hyperparameter tuning...")
            grid_search.fit(X_train, train_labels)
            
            # Get best model
            best_model = grid_search.best_estimator_
            logger.info(f"Best parameters found: {grid_search.best_params_}")
            
            # Evaluate model
            self._evaluate_model(best_model, X_test, test_labels, features)
            
            return best_model
            
        except Exception as e:
            logger.error(f"Error training fingerprinting model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def train_anomaly_model(self, features: np.ndarray) -> Optional[IsolationForest]:
        """
        Train an Isolation Forest model for anomaly detection.
        
        Args:
            features: Feature array
            
        Returns:
            Trained IsolationForest or None if training fails
        """
        try:
            logger.info("Training Isolation Forest model for anomaly detection...")
            
            anomaly_detector = IsolationForest(
                contamination=self.config.contamination,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )
            
            anomaly_detector.fit(features)
            logger.info("Anomaly detection model trained successfully.")
            
            return anomaly_detector
            
        except Exception as e:
            logger.error(f"Error training anomaly detection model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None
    
    def _get_param_grid(self) -> Dict[str, List[Any]]:
        """
        Define hyperparameter grid for model tuning.
        
        Returns:
            Dictionary of parameter grid
        """
        return {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    
    def _evaluate_model(self, model: RandomForestClassifier, X_test: np.ndarray, 
                        y_test: List[str], all_features: np.ndarray) -> None:
        """
        Evaluate model performance with multiple metrics.
        
        Args:
            model: Trained model
            X_test: Test feature data
            y_test: Test labels
            all_features: All features for cross-validation
        """
        # Test set evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        logger.info(f"Classification accuracy: {accuracy:.2f}%")
        
        # Detailed classification report
        logger.info("Classification Report:")
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")
        
        # Cross-validation
        all_labels = self.data_loader.generate_simulated_labels(len(all_features))
        skf = StratifiedKFold(n_splits=min(5, min(Counter(all_labels).values())))
        
        try:
            cv_scores = cross_val_score(model, all_features, all_labels, cv=skf)
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean cross-validation score: {np.mean(cv_scores) * 100:.2f}%")
        except Exception as e:
            logger.warning(f"Error during cross-validation: {e}")


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
            logger.info(f"Model saved to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model to {filename}: {e}")
            return False


def parse_arguments() -> ModelConfig:
    """
    Parse command line arguments.
    
    Returns:
        ModelConfig object with parsed arguments
    """
    parser = argparse.ArgumentParser(description="Train ML models for RF fingerprinting and anomaly detection.")
    
    parser.add_argument("-i", "--input", type=str, default="collected_iq_data.csv",
                        help="Path to the collected data file (default: collected_iq_data.csv)")
    parser.add_argument("-a", "--anomaly", type=str, default="anomaly_detection_model.pkl",
                        help="Path to the output anomaly model file (default: anomaly_detection_model.pkl)")
    parser.add_argument("-f", "--fingerprint", type=str, default="rf_fingerprinting_model.pkl",
                        help="Path to the output fingerprinting model file (default: rf_fingerprinting_model.pkl)")
    parser.add_argument("-v", "--verbose", type=int, choices=[0, 1, 2, 3], default=1,
                        help="Verbosity level (0-3, default: 1)")
    parser.add_argument("-j", "--jobs", type=int, default=-1,
                        help="Number of parallel jobs (-1 for all cores, default: -1)")
    
    args = parser.parse_args()
    
    # Configure logging based on verbosity
    log_levels = {
        0: logging.ERROR,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG
    }
    logger.setLevel(log_levels[args.verbose])
    
    # Return configuration object
    return ModelConfig(
        input_file=args.input,
        fingerprint_model_file=args.fingerprint,
        anomaly_model_file=args.anomaly,
        n_jobs=args.jobs,
        verbose=args.verbose
    )


def main():
    """Main entry point for the application."""
    try:
        # Parse arguments
        config = parse_arguments()
        
        # Log configuration
        logger.info(f"Loading data from {config.input_file}...")
        logger.info(f"Fingerprint model will be saved to {config.fingerprint_model_file}")
        logger.info(f"Anomaly model will be saved to {config.anomaly_model_file}")
        
        # Load data
        try:
            features = DataLoader.load_from_csv(config.input_file)
            logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features each")
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Sample features (first 5):\n{features[:5]}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return 1
            
        # Initialize trainer
        trainer = ModelTrainer(config)
        
        # Train models
        fingerprint_model = trainer.train_fingerprinting_model(features)
        anomaly_model = trainer.train_anomaly_model(features)
        
        # Save models
        model_saver = ModelSaver()
        
        if fingerprint_model is not None:
            model_saver.save_model(fingerprint_model, config.fingerprint_model_file)
        else:
            logger.error("Failed to train fingerprinting model")
        
        if anomaly_model is not None:
            model_saver.save_model(anomaly_model, config.anomaly_model_file)
        else:
            logger.error("Failed to train anomaly detection model")
            
        logger.info("Model training and saving completed successfully.")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
        return 130  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())
