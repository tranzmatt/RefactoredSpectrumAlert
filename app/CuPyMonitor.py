#!/usr/bin/env python3
"""
SpectrumAlert Monitor - SDR-based spectrum monitoring with anomaly detection and RF fingerprinting (GPU-accelerated).
"""

import argparse
import datetime
import sys
from typing import Optional, Any

import joblib
import cupy as np
import numpy as cpu_np
from sklearn.ensemble import RandomForestClassifier, IsolationForest

from spectrum_utils import (
    Config, ConfigReader, GPSHandler, DeviceIdentifier, 
    MQTTHandler, SDRHandler, FeatureExtractor
)


class SpectrumMonitor:
    def __init__(self, config: Config, anomaly_model: IsolationForest, fingerprint_model: RandomForestClassifier):
        self.config = config
        self.anomaly_model = anomaly_model
        self.fingerprint_model = fingerprint_model
        self.sdr_handler = None
        self.mqtt_handler = MQTTHandler()
        self.device_name = DeviceIdentifier.get_device_name()
        self.known_features = []
        self.expected_num_features = self._get_expected_feature_count(anomaly_model)

    def initialize(self):
        self.sdr_handler = SDRHandler(self.config.sdr_type, self.config.gain_value, self.config.sample_rate)
        self.sdr_handler.open()
        self.sdr_handler.setup_stream()

        mqtt_client, mqtt_topic = self.mqtt_handler.setup_client()
        if mqtt_client is None:
            raise RuntimeError("MQTT client initialization failed.")

    def run_monitoring(self):
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
        for band_start, band_end in self.config.ham_bands:
            current_freq = band_start
            print(f"Scanning band: {band_start/1e6:.3f}-{band_end/1e6:.3f} MHz")

            while current_freq <= band_end:
                self._process_frequency(current_freq)
                current_freq += self.config.freq_step

    def _process_frequency(self, frequency: float):
        for _ in range(self.config.runs_per_freq):
            self.sdr_handler.set_frequency(frequency)
            iq_samples = self.sdr_handler.read_samples()

            if iq_samples is None:
                continue

            features = FeatureExtractor.extract_features(
                iq_samples, target_num_features=self.expected_num_features
            )
            signal_strength_db = FeatureExtractor.calculate_signal_strength(iq_samples)

            if signal_strength_db < self.config.min_db:
                continue

            self._check_for_anomaly(frequency, features, signal_strength_db)
            self._update_fingerprinting_model(features)

    def _check_for_anomaly(self, frequency: float, features: list, signal_strength_db: float):
        features_cpu = cpu_np.asarray(features)
        is_anomaly = self.anomaly_model.predict([features_cpu])[0] == -1

        if is_anomaly:
            freq_mhz = frequency / 1e6
            print(f"Anomaly detected at {freq_mhz:.2f} MHz at {signal_strength_db:.2f} dB")

            freq_data = {
                'anomaly_freq_mhz': freq_mhz,
                'signal_strength': signal_strength_db
            }

            detection_time = datetime.datetime.utcnow().isoformat() + "Z"
            latitude, longitude, altitude = GPSHandler.get_coordinates()

            mqtt_payload = {
                "device": self.device_name,
                "detection_time": detection_time,
                "gps": {"lat": latitude, "lon": longitude, "alt": altitude},
                "data": freq_data
            }

            self.mqtt_handler.publish_message(self.mqtt_handler.topic, mqtt_payload)

    def _update_fingerprinting_model(self, features: list):
        self.known_features.append(features)

        if len(self.known_features) > 1:
            try:
                labels = ["Device" for _ in self.known_features]
                features_cpu = cpu_np.asarray(self.known_features)
                self.fingerprint_model.fit(features_cpu, labels)
            except Exception as e:
                print(f"Error updating fingerprinting model: {e}")

    def _cleanup(self):
        self.mqtt_handler.disconnect()
        if self.sdr_handler:
            self.sdr_handler.close()
        print("Cleanup complete.")

    @staticmethod
    def _get_expected_feature_count(model: Any) -> int:
        try:
            return model.n_features_in_
        except AttributeError:
            try:
                return model.estimators_[0].n_features_in_
            except (AttributeError, IndexError):
                return 9


def load_model(model_file: str, model_type: str) -> Optional[Any]:
    try:
        model = joblib.load(model_file)
        print(f"{model_type.capitalize()} model loaded from {model_file}")
        return model
    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        return None


def parse_arguments():
    parser = argparse.ArgumentParser(description="Spectrum Monitoring with SDR (GPU version).")
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file (default: config.ini)")
    parser.add_argument("-a", "--amodel", type=str, default="anomaly_detection_model.pkl",
                        help="Path to the anomaly detection model (default: anomaly_detection_model.pkl)")
    parser.add_argument("-r", "--rfmodel", type=str, default="rf_fingerprinting_model.pkl",
                        help="Path to the RF fingerprinting model (default: rf_fingerprinting_model.pkl)")
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()

        print(f"Using config file: {args.config}")
        print(f"Using anomaly model: {args.amodel}")
        print(f"Using RF fingerprinting model: {args.rfmodel}")

        config = ConfigReader.read_config(args.config)
        print(f"Configured for {len(config.ham_bands)} frequency bands")

        anomaly_model = load_model(args.amodel, "anomaly")
        fingerprint_model = load_model(args.rfmodel, "fingerprint")

        if anomaly_model is None or fingerprint_model is None:
            print("Failed to load required models. Exiting.")
            return 1

        monitor = SpectrumMonitor(config, anomaly_model, fingerprint_model)
        monitor.run_monitoring()
        return 0

    except KeyboardInterrupt:
        print("Program interrupted by user.")
        return 130
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

