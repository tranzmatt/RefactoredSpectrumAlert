#!/usr/bin/env python3
"""
SpectrumAlert Monitor - SDR-based spectrum monitoring with anomaly detection and RF fingerprinting.
"""

import argparse
import datetime
import sys
from typing import Optional, Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Import common utilities
from spectrum_utils import (
    Config, ConfigReader, GPSHandler, DeviceIdentifier, 
    MQTTHandler, SDRHandler, FeatureExtractor
)


class SpectrumMonitor:
    """Main class for spectrum monitoring with anomaly detection and fingerprinting."""
    
    def __init__(self, config: Config, anomaly_model: IsolationForest, fingerprint_model: RandomForestClassifier):
        """
        Initialize the spectrum monitor.
        
        Args:
            config: Configuration settings
            anomaly_model: Pre-trained anomaly detection model
            fingerprint_model: Pre-trained fingerprinting model
        """
        self.config = config
        self.anomaly_model = anomaly_model
        self.fingerprint_model = fingerprint_model
        self.sdr_handler = None
        self.mqtt_handler = MQTTHandler()
        self.device_name = DeviceIdentifier.get_device_name()
        self.known_features = []  # Storage for collected features
        
        # Get number of features the anomaly model expects
        self.expected_num_features = self._get_expected_feature_count(anomaly_model)
            
    def initialize(self):
        """Initialize handlers and connections."""
        # Set up SDR
        self.sdr_handler = SDRHandler(self.config.sdr_type, self.config.gain_value, self.config.sample_rate)
        self.sdr_handler.open()
        self.sdr_handler.setup_stream()
        
        # Set up MQTT
        mqtt_client, mqtt_topic = self.mqtt_handler.setup_client()
        if mqtt_client is None:
            raise RuntimeError("MQTT client initialization failed.")
        
    def run_monitoring(self):
        """Run continuous spectrum monitoring."""
        try:
            self.initialize()
            
            while True:
                self._scan_bands()
                    
        except KeyboardInterrupt:
            print("Monitoring stopped by user.")
        except Exception as e:
            print(f"Error during monitoring: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
            
    def _scan_bands(self):
        """Scan through all configured frequency bands."""
        for band_start, band_end in self.config.ham_bands:
            current_freq = band_start
            print(f"Scanning band: {band_start/1e6:.3f}-{band_end/1e6:.3f} MHz")
            
            while current_freq <= band_end:
                self._process_frequency(current_freq)
                current_freq += self.config.freq_step
                
    def _process_frequency(self, frequency: float):
        """
        Process a single frequency by collecting and analyzing samples.
        
        Args:
            frequency: Frequency to monitor in Hz
        """
        for _ in range(self.config.runs_per_freq):
            self.sdr_handler.set_frequency(frequency)
            iq_samples = self.sdr_handler.read_samples()
            
            if iq_samples is None:
                continue
                
            # Extract features and calculate signal strength
            features = FeatureExtractor.extract_features(
                iq_samples, target_num_features=self.expected_num_features
            )
            signal_strength_db = FeatureExtractor.calculate_signal_strength(iq_samples)
            
            # Skip if signal is too weak
            if signal_strength_db < self.config.min_db:
                continue
                
            # Check for anomaly
            self._check_for_anomaly(frequency, features, signal_strength_db)
            
            # Update fingerprinting model with new data
            self._update_fingerprinting_model(features)
    
    def _check_for_anomaly(self, frequency: float, features: list, signal_strength_db: float):
        """
        Check if the features represent an anomaly and publish if true.
        
        Args:
            frequency: Current frequency in Hz
            features: Extracted features
            signal_strength_db: Signal strength in dB
        """
        is_anomaly = self.anomaly_model.predict([features])[0] == -1
        
        if is_anomaly:
            freq_mhz = frequency / 1e6
            print(f"Anomaly detected at {freq_mhz:.2f} MHz at {signal_strength_db:.2f} dB")
            
            # Prepare anomaly data
            freq_data = {
                'anomaly_freq_mhz': freq_mhz,
                'signal_strength': signal_strength_db
            }
            
            # Get detection time and GPS coordinates
            detection_time = datetime.datetime.utcnow().isoformat() + "Z"  # Add "Z" for UTC
            latitude, longitude, altitude = GPSHandler.get_coordinates()
            
            # Create payload
            mqtt_payload = {
                "device": self.device_name,
                "detection_time": detection_time,
                "gps": {"lat": latitude, "lon": longitude, "alt": altitude},
                "data": freq_data
            }
            
            # Publish to MQTT
            self.mqtt_handler.publish_message(self.mqtt_handler.topic, mqtt_payload)
    
    def _update_fingerprinting_model(self, features: list):
        """
        Update the fingerprinting model with new feature data.
        
        Args:
            features: Extracted features
        """
        # Add features to known features list
        self.known_features.append(features)
        
        # Only update model if we have enough data
        if len(self.known_features) > 1:
            try:
                # Create labels (all from same device in this implementation)
                labels = ["Device" for _ in self.known_features]
                
                # Update model
                self.fingerprint_model.fit(self.known_features, labels)
            except Exception as e:
                print(f"Error updating fingerprinting model: {e}")
    
    def _cleanup(self):
        """Clean up resources."""
        self.mqtt_handler.disconnect()
        
        if self.sdr_handler:
            self.sdr_handler.close()
            
        print("Cleanup complete.")
            
    @staticmethod
    def _get_expected_feature_count(model: Any) -> int:
        """
        Determine the number of features expected by the model.
        
        Args:
            model: Trained model
            
        Returns:
            Number of features the model expects
        """
        try:
            # First try n_features_in_ attribute (sklearn 0.24+)
            return model.n_features_in_
        except AttributeError:
            try:
                # Then try to get from estimators (IsolationForest)
                return model.estimators_[0].n_features_in_
            except (AttributeError, IndexError):
                # Default value
                return 9


def load_model(model_file: str, model_type: str) -> Optional[Any]:
    """
    Load a pre-trained ML model.
    
    Args:
        model_file: Path to the model file
        model_type: Type of model ('anomaly' or 'fingerprint')
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        model = joblib.load(model_file)
        print(f"{model_type.capitalize()} model loaded from {model_file}")
        return model
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="Spectrum Monitoring with SDR.")
    
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file (default: config.ini)")
    parser.add_argument("-a", "--amodel", type=str, default="anomaly_detection_model.pkl",
                        help="Path to the anomaly detection model (default: anomaly_detection_model.pkl)")
    parser.add_argument("-r", "--rfmodel", type=str, default="rf_fingerprinting_model.pkl",
                        help="Path to the RF fingerprinting model (default: rf_fingerprinting_model.pkl)")
    
    return parser.parse_args()


def main():
    """Main entry point for the application."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Display configuration
        print(f"Using config file: {args.config}")
        print(f"Using anomaly model: {args.amodel}")
        print(f"Using RF fingerprinting model: {args.rfmodel}")
        
        # Read configuration
        config = ConfigReader.read_config(args.config)
        print(f"Configured for {len(config.ham_bands)} frequency bands")
        
        # Load models
        anomaly_model = load_model(args.amodel, "anomaly")
        fingerprint_model = load_model(args.rfmodel, "fingerprint")
        
        if anomaly_model is None or fingerprint_model is None:
            print("Failed to load required models. Exiting.")
            return 1
        
        # Create and run monitor
        monitor = SpectrumMonitor(config, anomaly_model, fingerprint_model)
        monitor.run_monitoring()
        
        return 0
        
    except KeyboardInterrupt:
        print("Program interrupted by user.")
        return 130  # Standard exit code for SIGINT
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
