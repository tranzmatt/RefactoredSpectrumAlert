#!/usr/bin/env python3
"""
SpectrumAlert MiniTrainer - Lightweight ML model trainer for RF fingerprinting and anomaly detection.
Adapted for CuPy (GPU-accelerated NumPy).
"""

import argparse
import os
import sys
from collections import Counter
from typing import List, Optional, Tuple, Any

import joblib
import cupy as np  # CuPy used in place of NumPy
import numpy as cpu_np  # For CPU conversions required by sklearn
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold

from spectrum_utils import (
    CSVDataHandler, safe_float_conversion
)

DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_CONTAMINATION = 0.05
DEFAULT_DEVICE_COUNT = 5

class DataHandler:
    @staticmethod
    def load_from_csv(filename: str) -> np.ndarray:
        if not os.path.exists(filename):
            raise FileNotFoundError(f"File {filename} not found.")

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
                feature_values = [safe_float_conversion(value) for value in row[1:]]
                features.append(feature_values)
            except Exception as e:
                print(f"Warning: Error parsing row {row}: {e}")

        if not features:
            raise ValueError("No valid data found in the CSV file.")

        return np.asarray(cpu_np.array(features))

    @staticmethod
    def generate_labels(data_length: int, device_count: int) -> List[str]:
        return [f"Device_{i % device_count}" for i in range(data_length)]


class LiteModelTrainer:
    def __init__(self, device_count=DEFAULT_DEVICE_COUNT, 
                 test_size=DEFAULT_TEST_SIZE, 
                 random_state=DEFAULT_RANDOM_STATE,
                 contamination=DEFAULT_CONTAMINATION):
        self.device_count = device_count
        self.test_size = test_size
        self.random_state = random_state
        self.contamination = contamination
        self.data_handler = DataHandler()

    def train_models(self, features: np.ndarray) -> Tuple[Optional[RandomForestClassifier], Optional[IsolationForest]]:
        if len(features) < 2:
            print("Not enough data to train the model.")
            return None, None

        try:
            labels = self.data_handler.generate_labels(len(features), self.device_count)
            fingerprint_model = self._train_fingerprinting_model(features, labels)
            anomaly_model = self._train_anomaly_model(features)
            return fingerprint_model, anomaly_model
        except Exception as e:
            print(f"Error during model training: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def _train_fingerprinting_model(self, features: np.ndarray, labels: List[str]) -> RandomForestClassifier:
        features_cpu = features.get()
        X_train, X_test, y_train, y_test = train_test_split(
            features_cpu, labels, test_size=self.test_size, random_state=self.random_state
        )

        class_counts = Counter(y_train)
        min_samples_per_class = min(class_counts.values())
        max_cv_splits = min(3, min_samples_per_class)

        print(f"Using {max_cv_splits}-fold cross-validation (lite version).")

        model = RandomForestClassifier(
            n_estimators=50, max_depth=10, random_state=self.random_state
        )

        print("Training the RF fingerprinting model (lite version)...")
        model.fit(X_train, y_train)

        self._evaluate_model(model, X_test, y_test, features_cpu, labels, max_cv_splits)
        return model

    def _train_anomaly_model(self, features: np.ndarray) -> IsolationForest:
        features_cpu = features.get()
        print("Training the IsolationForest model for anomaly detection (lite version)...")

        model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=50
        )

        model.fit(features_cpu)
        print("Anomaly detection model trained successfully.")
        return model

    def _evaluate_model(self, model: RandomForestClassifier, X_test: cpu_np.ndarray, 
                        y_test: List[str], features: cpu_np.ndarray, labels: List[str], 
                        cv_splits: int) -> None:
        y_pred = model.predict(X_test)
        print(f"Classification accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        skf = StratifiedKFold(n_splits=cv_splits)
        try:
            cv_scores = cross_val_score(model, features, labels, cv=skf)
            print(f"Cross-validation scores: {cv_scores}")
            print(f"Mean cross-validation score: {cpu_np.mean(cv_scores) * 100:.2f}%")
        except Exception as e:
            print(f"Warning: Error during cross-validation: {e}")


class ModelSaver:
    @staticmethod
    def save_model(model: Any, filename: str) -> bool:
        try:
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
    parser = argparse.ArgumentParser(
        description="Lightweight ML trainer for RF fingerprinting and anomaly detection (GPU version)."
    )
    parser.add_argument("-i", "--input", type=str, default="collected_data_lite.csv",
                        help="Path to the collected lite data file")
    parser.add_argument("-a", "--anomaly", type=str, default="anomaly_detection_model_lite.pkl",
                        help="Path to save the anomaly detection model")
    parser.add_argument("-f", "--fingerprint", type=str, default="rf_fingerprinting_model_lite.pkl",
                        help="Path to save the fingerprinting model")
    parser.add_argument("-d", "--devices", type=int, default=DEFAULT_DEVICE_COUNT,
                        help="Number of devices to simulate")
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()

        print(f"Loading data from {args.input}...")
        print(f"Fingerprint model will be saved to {args.fingerprint}")
        print(f"Anomaly model will be saved to {args.anomaly}")
        print(f"Simulating {args.devices} devices")

        try:
            data_handler = DataHandler()
            features = data_handler.load_from_csv(args.input)
            print(f"Loaded {len(features)} samples with {features.shape[1]} features each")
            print(f"Sample features (first 5):\n{features[:5]}")
        except Exception as e:
            print(f"Error loading data: {e}")
            return 1

        trainer = LiteModelTrainer(device_count=args.devices)
        fingerprint_model, anomaly_model = trainer.train_models(features)

        print("Saving models...")
        saver = ModelSaver()

        if fingerprint_model:
            saver.save_model(fingerprint_model, args.fingerprint)
        if anomaly_model:
            saver.save_model(anomaly_model, args.anomaly)

        print("Training and saving completed successfully.")
        return 0

    except KeyboardInterrupt:
        print("Training interrupted by user.")
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

