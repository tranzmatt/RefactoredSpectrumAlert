#!/usr/bin/env python3
"""
SpectrumAlert MiniData - Lightweight SDR-based spectrum data collection (GPU-accelerated).
"""

import argparse
import sys
import time
from typing import List, Optional

import cupy as np
import numpy as cpu_np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

from spectrum_utils import (
    Config, ConfigReader, SDRHandler, FeatureExtractor, CSVDataHandler
)

LITE_SAMPLE_SIZE = 128 * 1024


class DataCollector:
    def __init__(self, config: Config, output_file: str):
        self.config = config
        self.sdr_handler = None
        self.csv_handler = CSVDataHandler(
            output_file,
            header=['Frequency', 'Mean_Amplitude', 'Std_Amplitude']
        )

    def initialize(self):
        self.sdr_handler = SDRHandler(
            self.config.sdr_type,
            self.config.gain_value,
            self.config.sample_rate
        )
        self.sdr_handler.open()
        self.sdr_handler.setup_stream()

    def collect_training_data(self) -> List[List[float]]:
        pca_training_data = []

        for band_start, band_end in self.config.ham_bands:
            self.sdr_handler.set_frequency(band_start)
            iq_samples = self.sdr_handler.read_samples(LITE_SAMPLE_SIZE)

            if iq_samples is not None:
                features = self._extract_lite_features(iq_samples)
                pca_training_data.append(features)
                print(f"Collected PCA training data at {band_start / 1e6:.3f} MHz")

        if not pca_training_data:
            raise RuntimeError("Failed to collect training data")

        return pca_training_data

    def train_pca(self, training_data: List[List[float]]) -> PCA:
        n_components = min(2, len(training_data[0]), len(training_data))
        pca = PCA(n_components=n_components)
        pca.fit(cpu_np.asarray(training_data))
        return pca

    def gather_data(self, duration_minutes: float):
        try:
            self.initialize()
            training_data = self.collect_training_data()
            pca = self.train_pca(training_data)

            anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
            anomaly_detector.fit(cpu_np.asarray(training_data))

            self.sdr_handler.close()
            self.initialize()

            start_time = time.time()
            duration_seconds = duration_minutes * 60

            while time.time() - start_time < duration_seconds:
                self._scan_frequency_bands(pca)

        except KeyboardInterrupt:
            print("Data collection interrupted by user.")
        except Exception as e:
            print(f"Error during data collection: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.sdr_handler:
                self.sdr_handler.close()
                print("Closed SDR device.")

    def _scan_frequency_bands(self, pca: PCA):
        for band_start, band_end in self.config.ham_bands:
            current_freq = band_start
            while current_freq <= band_end:
                self._process_frequency(current_freq, pca)
                current_freq += self.config.freq_step

    def _process_frequency(self, frequency: float, pca: PCA):
        run_features = []

        for _ in range(self.config.runs_per_freq):
            self.sdr_handler.set_frequency(frequency)
            iq_samples = self.sdr_handler.read_samples(LITE_SAMPLE_SIZE)

            if iq_samples is not None:
                lite_features = self._extract_lite_features(iq_samples)
                run_features.append(lite_features)

        if run_features:
            avg_features = np.mean(np.asarray(run_features), axis=0).get()
            reduced_features = pca.transform([avg_features])[0]
            data = [frequency] + reduced_features.tolist()
            self.csv_handler.write_row(data)

    def _extract_lite_features(self, iq_data: np.ndarray) -> List[float]:
        I = np.real(iq_data)
        Q = np.imag(iq_data)
        amplitude = np.sqrt(I ** 2 + Q ** 2)
        mean_amplitude = np.mean(amplitude).item()
        std_amplitude = np.std(amplitude).item()
        return [mean_amplitude, std_amplitude]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Lightweight Spectrum Data Collection with SDR.")
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file (default: config.ini)")
    parser.add_argument("-d", "--duration", type=float, default=10,
                        help="Duration in minutes (default: 10)")
    parser.add_argument("-o", "--output", type=str, default="collected_data_lite.csv",
                        help="Path to the output file (default: collected_data_lite.csv)")
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()

        print(f"Using config file: {args.config}")
        print(f"Monitoring duration: {args.duration} minutes")
        print(f"Saving to: {args.output}")

        config = ConfigReader.read_config(args.config)
        print(f"Configured for {len(config.ham_bands)} frequency bands")

        collector = DataCollector(config, args.output)

        print(f"Starting data collection for {args.duration} minutes...")
        collector.gather_data(args.duration)
        print("Data collection completed.")
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

