#!/usr/bin/env python3
"""
SpectrumAlert MiniMonitor - Lightweight SDR-based spectrum monitoring with anomaly detection.
"""

import argparse
import datetime
import json
import sys
import time
from typing import Any, Optional

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

# Import common utilities
from spectrum_utils import (
    Config, ConfigReader, GPSHandler, DeviceIdentifier, 
    MQTTHandler, SDRHandler, FeatureExtractor
)


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="Spectrum Monitoring with SDR.")
    
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file (default: onfig.ini)")
    parser.add_argument("-a", "--amodel", type=str, default="anomaly_detection_model_lite.pkl",
                        help="Path to the anomaly detection model (default: anomaly_detection_model_lite.pkl)")
    
    return parser.parse_args()


def load_anomaly_detection_model(model_file: str) -> Optional[IsolationForest]:
    """
    Load the pre-trained anomaly detection model.
    
    Args:
        model_file: Path to the model file
        
    Returns:
        Loaded model or None if loading fails
    """
    try:
        model = joblib.load(model_file)
        print(f"Anomaly detection model loaded from {model_file}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_expected_feature_count(model: Any) -> int:
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
            return 2


def monitor_spectrum(config: Config, anomaly_model: IsolationForest):
    """
    Monitor the spectrum for anomalies.
    
    Args:
        config: Configuration settings
        anomaly_model: Pre-trained anomaly detection model
    """
    # Initialize handlers
    mqtt_handler = MQTTHandler()
    mqtt_client, mqtt_topic = mqtt_handler.setup_client()
    if mqtt_client is None:
        print("❌ MQTT client initialization failed. Exiting...")
        sys.exit(1)
        
    sdr_handler = SDRHandler(config.sdr_type, config.gain_value, config.sample_rate)
    device_name = DeviceIdentifier.get_device_name()
    
    try:
        # Get the number of features the model expects
        expected_num_features = get_expected_feature_count(anomaly_model)
        print(f"Model expects {expected_num_features} features")
        
        # Open SDR and setup stream
        sdr_handler.open()
        sdr_handler.setup_stream()
        
        # Continuously monitor frequency bands
        while True:
            for band_start, band_end in config.ham_bands:
                current_freq = band_start
                print(f"Scanning band: {band_start/1e6:.3f}-{band_end/1e6:.3f} MHz")
                
                while current_freq <= band_end:
                    print(f"Scanning band: {current_freq/1e6:.3f} MHz")
                    for _ in range(config.runs_per_freq):
                        # Set frequency and read samples
                        sdr_handler.set_frequency(current_freq)
                        iq_samples = sdr_handler.read_samples()
                        
                        if iq_samples is not None:
                            # Extract features
                            features = FeatureExtractor.extract_features(
                                iq_samples, target_num_features=expected_num_features
                            )
                            signal_strength_db = FeatureExtractor.calculate_signal_strength(iq_samples)
                            
                            # Skip if signal is too weak
                            if signal_strength_db < config.min_db:
                                continue
                                
                            # Detect anomalies
                            is_anomaly = anomaly_model.predict([features])[0] == -1
                            if is_anomaly:
                                # Prepare anomaly data
                                freq_data = {
                                    'anomaly_freq_mhz': (current_freq / 1e6),
                                    'signal_strength': signal_strength_db
                                }
                                
                                # Get detection time and GPS coordinates
                                detection_time = datetime.datetime.utcnow().isoformat() + "Z"
                                latitude, longitude, altitude = GPSHandler.get_coordinates()
                                
                                # Create and publish message
                                mqtt_payload = {
                                    "device": device_name,
                                    "detection_time": detection_time,
                                    "gps": {"lat": latitude, "lon": longitude, "alt": altitude},
                                    "data": freq_data
                                }
                                
                                mqtt_handler.publish_message(mqtt_topic, mqtt_payload)
                                print(f"Anomaly at {current_freq / 1e6:.2f} MHz, signal strength: {signal_strength_db:.1f} dB")
                        
                    current_freq += config.freq_step
                    
    except KeyboardInterrupt:
        print("Monitoring stopped by user.")
    except Exception as e:
        print(f"Error during monitoring: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Cleanup
        mqtt_handler.disconnect()
        sdr_handler.close()
        print("Cleanup complete.")


def main():
    """Main entry point for the application."""
    try:
        # Parse arguments
        args = parse_arguments()
        print(f"Using config file: {args.config}")
        print(f"Using anomaly model: {args.amodel}")
        
        # Read configuration
        config = ConfigReader.read_config(args.config)
        print(f"Configured for {len(config.ham_bands)} frequency bands")
        
        # Load model
        anomaly_model = load_anomaly_detection_model(args.amodel)
        if anomaly_model is None:
            print("Failed to load anomaly detection model. Exiting.")
            return 1
        
        # Start monitoring
        monitor_spectrum(config, anomaly_model)
        
        return 0
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
