#!/usr/bin/env python3
"""
SpectrumAlert MiniData - Lightweight SDR-based spectrum data collection.
"""

import argparse
import sys
import time
from typing import List, Optional

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# Import common utilities
from spectrum_utils import (
    Config, ConfigReader, SDRHandler, FeatureExtractor, CSVDataHandler
)

# Lite version parameters
LITE_SAMPLE_SIZE = 128 * 1024  # Reduced sample size for Raspberry Pi


class DataCollector:
    """Handles data collection and processing."""
    
    def __init__(self, config: Config, output_file: str):
        """
        Initialize the data collector.
        
        Args:
            config: Configuration settings
            output_file: Path to output CSV file
        """
        self.config = config
        self.sdr_handler = None
        self.csv_handler = CSVDataHandler(
            output_file, 
            header=['Frequency', 'Mean_Amplitude', 'Std_Amplitude']
        )
        
    def initialize(self):
        """Initialize SDR handler."""
        self.sdr_handler = SDRHandler(
            self.config.sdr_type, 
            self.config.gain_value,
            self.config.sample_rate
        )
        self.sdr_handler.open()
        self.sdr_handler.setup_stream()
        
    def collect_training_data(self) -> List[List[float]]:
        """
        Collect initial data for PCA and anomaly detection training.
        
        Returns:
            List of feature lists for training
        """
        pca_training_data = []
        
        # Collect some initial data for training
        for band_start, band_end in self.config.ham_bands:
            self.sdr_handler.set_frequency(band_start)
            iq_samples = self.sdr_handler.read_samples(LITE_SAMPLE_SIZE)
            
            if iq_samples is not None:
                # Important: Use the same feature extraction method as in _extract_lite_features
                features = self._extract_lite_features(iq_samples)
                pca_training_data.append(features)
                print(f"Collected PCA training data at {band_start / 1e6:.3f} MHz")
        
        if not pca_training_data:
            raise RuntimeError("Failed to collect training data")
            
        return pca_training_data
    
    def train_pca(self, training_data: List[List[float]]) -> PCA:
        """
        Train a PCA model on the collected data.
        
        Args:
            training_data: Feature lists for training
            
        Returns:
            Trained PCA model
        """
        # Determine dimensionality
        n_components = min(2, len(training_data[0]), len(training_data))
        
        # Initialize and train PCA
        pca = PCA(n_components=n_components)
        pca.fit(training_data)
        
        return pca
    
    def gather_data(self, duration_minutes: float):
        """
        Gather data for the specified duration.
        
        Args:
            duration_minutes: Duration to collect data in minutes
        """
        try:
            # Initialize SDR
            self.initialize()
            
            # Collect training data and train PCA
            training_data = self.collect_training_data()
            pca = self.train_pca(training_data)
            
            # Initialize anomaly detector (optional)
            anomaly_detector = IsolationForest(contamination=0.05, random_state=42)
            anomaly_detector.fit(training_data)
            
            # Reset for continuous data collection
            self.sdr_handler.close()
            self.initialize()
            
            # Monitor bands
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
        """
        Scan through all configured frequency bands.
        
        Args:
            pca: Trained PCA model for dimensionality reduction
        """
        for band_start, band_end in self.config.ham_bands:
            current_freq = band_start
            while current_freq <= band_end:
                self._process_frequency(current_freq, pca)
                current_freq += self.config.freq_step
    
    def _process_frequency(self, frequency: float, pca: PCA):
        """
        Process a single frequency by collecting and analyzing samples.
        
        Args:
            frequency: Frequency to sample in Hz
            pca: Trained PCA model
        """
        run_features = []
        
        # Multiple runs per frequency for better reliability
        for _ in range(self.config.runs_per_freq):
            self.sdr_handler.set_frequency(frequency)
            iq_samples = self.sdr_handler.read_samples(LITE_SAMPLE_SIZE)
            
            if iq_samples is not None:
                # Extract only the features we need for the lite version
                # IMPORTANT: Use the same feature extraction method as in collect_training_data
                lite_features = self._extract_lite_features(iq_samples)
                run_features.append(lite_features)
        
        if run_features:
            # Average features across runs
            avg_features = np.mean(run_features, axis=0)
            
            # Apply PCA for dimensionality reduction
            reduced_features = pca.transform([avg_features])[0]
            
            # Prepare and save data
            data = [frequency] + reduced_features.tolist()
            #print(f"Data is {data}")
            self.csv_handler.write_row(data)
    
    def _extract_lite_features(self, iq_data: np.ndarray) -> List[float]:
        """
        Extract a minimal set of features for the lite version.
        
        Args:
            iq_data: Complex IQ data samples
            
        Returns:
            List of extracted features
        """
        # Extract just the basic amplitude features for efficiency
        I = np.real(iq_data)
        Q = np.imag(iq_data)
        amplitude = np.sqrt(I ** 2 + Q ** 2)
        
        # Basic amplitude statistics (for lite version)
        mean_amplitude = np.mean(amplitude)
        std_amplitude = np.std(amplitude)
        
        # Return only 2 features
        return [mean_amplitude, std_amplitude]


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="Lightweight Spectrum Data Collection with SDR.")
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file (default: config.ini)")
    parser.add_argument("-d", "--duration", type=float, default=10,
                        help="Duration in minutes (default: 10)")
    parser.add_argument("-o", "--output", type=str, default="collected_data_lite.csv",
                        help="Path to the output file (default: collected_data_lite.csv)")

    return parser.parse_args()


def main():
    """Main entry point for the application."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Display configuration
        print(f"Using config file: {args.config}")
        print(f"Monitoring duration: {args.duration} minutes")
        print(f"Saving to: {args.output}")
        
        # Read configuration
        config = ConfigReader.read_config(args.config)
        print(f"Configured for {len(config.ham_bands)} frequency bands")
        
        # Create data collector
        collector = DataCollector(config, args.output)
        
        # Start data collection
        print(f"Starting data collection for {args.duration} minutes...")
        collector.gather_data(args.duration)
        print("Data collection completed.")
        
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
