#!/usr/bin/env python3
"""
SpectrumAlert ModelTrainer - Train ML models for RF fingerprinting and anomaly detection (GPU-accelerated).
"""

import argparse
import logging
import os
import sys
from collections import Counter
from typing import Dict, List, Optional, Tuple, Any

import joblib
import cupy as np
import numpy as cpu_np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold

from spectrum_utils import (
    CSVDataHandler, safe_float_conversion
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ModelTrainer')


class ModelConfig:
    def __init__(self, input_file: str, fingerprint_model_file: str, anomaly_model_file: str,
                 test_size: float = 0.2, random_state: int = 42, contamination: float = 0.05,
                 n_jobs: int = -1, verbose: int = 1):
        self.input_file = input_file
        self.fingerprint_model_file = fingerprint_model_file
        self.anomaly_model_file = anomaly_model_file
        self.test_size = test_size
        self.random_state = random_state
        self.contamination = contamination
        self.n_jobs = n_jobs
        self.verbose = verbose


class DataLoader:
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
                logger.warning(f"Skipping invalid row: {row}")
                continue
            try:
                feature_values = [safe_float_conversion(value) for value in row[1:]]
                features.append(feature_values)
            except Exception as e:
                logger.warning(f"Error parsing row {row}: {e}")

        if not features:
            raise ValueError("No valid data found in the CSV file.")

        return np.asarray(cpu_np.array(features))

    @staticmethod
    def split_data(features: np.ndarray, test_size: float, random_state: int) -> Tuple[np.ndarray, np.ndarray]:
        if len(features) < 2:
            raise ValueError("Not enough data to split into training and testing sets.")

        features_cpu = features.get()
        return train_test_split(features_cpu, test_size=test_size, random_state=random_state)

    @staticmethod
    def generate_simulated_labels(data_length: int, num_devices: int = 10) -> List[str]:
        return [f"Device_{i % num_devices}" for i in range(data_length)]


class ModelTrainer:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.data_loader = DataLoader()

    def train_fingerprinting_model(self, features: np.ndarray) -> Optional[RandomForestClassifier]:
        try:
            X_train, X_test = self.data_loader.split_data(features, self.config.test_size, self.config.random_state)
            train_labels = self.data_loader.generate_simulated_labels(len(X_train))
            test_labels = self.data_loader.generate_simulated_labels(len(X_test))

            class_counts = Counter(train_labels)
            min_samples_per_class = min(class_counts.values())
            max_cv_splits = min(5, min_samples_per_class)
            max_cv_splits = max(2, max_cv_splits)

            logger.info(f"Using {max_cv_splits}-fold cross-validation (based on smallest class size).")

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

            logger.info("Training Random Forest model with hyperparameter tuning...")
            grid_search.fit(X_train, train_labels)

            best_model = grid_search.best_estimator_
            logger.info(f"Best parameters found: {grid_search.best_params_}")

            self._evaluate_model(best_model, X_test, test_labels, features)
            return best_model

        except Exception as e:
            logger.error(f"Error training fingerprinting model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def train_anomaly_model(self, features: np.ndarray) -> Optional[IsolationForest]:
        try:
            logger.info("Training Isolation Forest model for anomaly detection...")
            features_cpu = features.get()

            anomaly_detector = IsolationForest(
                contamination=self.config.contamination,
                random_state=self.config.random_state,
                n_jobs=self.config.n_jobs
            )

            anomaly_detector.fit(features_cpu)
            logger.info("Anomaly detection model trained successfully.")
            return anomaly_detector

        except Exception as e:
            logger.error(f"Error training anomaly detection model: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            return None

    def _get_param_grid(self) -> Dict[str, List[Any]]:
        return {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }

    def _evaluate_model(self, model: RandomForestClassifier, X_test: cpu_np.ndarray,
                        y_test: List[str], all_features: np.ndarray) -> None:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred) * 100
        logger.info(f"Classification accuracy: {accuracy:.2f}%")
        report = classification_report(y_test, y_pred)
        logger.info(f"\n{report}")

        all_labels = self.data_loader.generate_simulated_labels(len(all_features))
        features_cpu = all_features.get()
        skf = StratifiedKFold(n_splits=min(5, min(Counter(all_labels).values())))

        try:
            cv_scores = cross_val_score(model, features_cpu, all_labels, cv=skf)
            logger.info(f"Cross-validation scores: {cv_scores}")
            logger.info(f"Mean cross-validation score: {cpu_np.mean(cv_scores) * 100:.2f}%")
        except Exception as e:
            logger.warning(f"Error during cross-validation: {e}")


class ModelSaver:
    @staticmethod
    def save_model(model: Any, filename: str) -> bool:
        try:
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

    log_levels = {
        0: logging.ERROR,
        1: logging.INFO,
        2: logging.DEBUG,
        3: logging.DEBUG
    }
    logger.setLevel(log_levels[args.verbose])

    return ModelConfig(
        input_file=args.input,
        fingerprint_model_file=args.fingerprint,
        anomaly_model_file=args.anomaly,
        n_jobs=args.jobs,
        verbose=args.verbose
    )


def main():
    try:
        config = parse_arguments()

        logger.info(f"Loading data from {config.input_file}...")
        logger.info(f"Fingerprint model will be saved to {config.fingerprint_model_file}")
        logger.info(f"Anomaly model will be saved to {config.anomaly_model_file}")

        try:
            features = DataLoader.load_from_csv(config.input_file)
            logger.info(f"Loaded {len(features)} samples with {features.shape[1]} features each")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Sample features (first 5):\n{features[:5]}")
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return 1

        trainer = ModelTrainer(config)
        fingerprint_model = trainer.train_fingerprinting_model(features)
        anomaly_model = trainer.train_anomaly_model(features)

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
        return 130
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        if logger.isEnabledFor(logging.DEBUG):
            import traceback
            logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    sys.exit(main())

