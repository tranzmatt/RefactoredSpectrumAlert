#!/usr/bin/env python3
"""
SpectrumAlert DataGathering - Multi-threaded SDR-based spectrum data collection.
"""

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Any

import numpy as np
import SoapySDR

from sklearn.decomposition import PCA

# Import common utilities
from spectrum_utils import (
    Config, ConfigReader, SDRHandler, FeatureExtractor, CSVDataHandler
)


class ThreadSafeCSVWriter:
    """Thread-safe CSV file writer."""
    
    def __init__(self, filename: str, feature_count: int):
        """
        Initialize the CSV writer.
        
        Args:
            filename: Path to the output CSV file
            feature_count: Number of features being extracted
        """
        self.lock = threading.Lock()
        self.csv_handler = CSVDataHandler(
            filename, 
            header=self._get_header_columns(feature_count)
        )
    
    def write_row(self, data: List[Any]):
        """
        Write a row of data to the CSV file in a thread-safe manner.
        
        Args:
            data: List of values to write
        """
        with self.lock:
            self.csv_handler.write_row(data)
    
    def _get_header_columns(self, feature_count: int) -> List[str]:
        """
        Generate the CSV header columns.
        
        Args:
            feature_count: Number of features
            
        Returns:
            List of column names
        """
        header = ['Frequency']
        
        # Feature column names
        feature_names = [
            'Mean_Amplitude', 'Std_Amplitude', 'Mean_FFT_Magnitude', 
            'Std_FFT_Magnitude', 'Skew_Amplitude', 'Kurt_Amplitude', 
            'Skew_Phase', 'Kurt_Phase', 'Cyclo_Autocorr',
            'Spectral_Entropy', 'PAPR', 'Band_Energy_Ratio'
        ]
        
        # Ensure we have the right number of column names for the features
        # If not enough names, add generic Feature_N names
        while len(feature_names) < feature_count:
            feature_names.append(f'Feature_{len(feature_names) + 1}')
            
        # If too many names, truncate
        feature_names = feature_names[:feature_count]
        
        return header + feature_names


class BandScanner:
    """Manages scanning of frequency bands with SDR devices."""
    
    def __init__(self, config: Config, csv_writer: ThreadSafeCSVWriter):
        """
        Initialize the band scanner.
        
        Args:
            config: Configuration settings
            csv_writer: Thread-safe CSV writer object
        """
        self.config = config
        self.csv_writer = csv_writer
        self.pca = None
        self.lock = threading.Lock()
        
    def initialize_pca(self):
        """
        Initialize and train the PCA model with data from the first device.
        
        Returns:
            Trained PCA model
        """
        print("Collecting PCA training data...")
        pca_training_data = []
        sdr_handler = None
        
        try:
            # Create scanner for first device
            sdr_handler = self._create_sdr_handler("0")  # Use first device
            
            # Collect samples from each band for PCA training
            for band_start, band_end in self.config.ham_bands:
                iq_samples = self._read_samples(sdr_handler, band_start)
                
                if iq_samples is not None:
                    features = FeatureExtractor.extract_features(iq_samples)
                    pca_training_data.append(features)
                    print(f"Collected PCA training data at {band_start / 1e6:.3f} MHz")
                    
            if not pca_training_data:
                raise RuntimeError("Failed to collect any PCA training data")
                
            # Train PCA model
            num_features = len(pca_training_data[0])
            n_components = min(8, len(pca_training_data), num_features)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(pca_training_data)
            print(f"PCA model trained with {len(pca_training_data)} samples, {n_components} components")
            
            return self.pca
            
        finally:
            if sdr_handler:
                sdr_handler.close()
    
    def scan_band(self, device_id: str, band_start: float, band_end: float):
        """
        Scan a frequency band using a specific device.
        
        Args:
            device_id: Device identifier
            band_start: Start frequency in Hz
            band_end: End frequency in Hz
        """
        sdr_handler = None
        thread_id = threading.get_ident()
        
        try:
            # Create and initialize scanner
            sdr_handler = self._create_sdr_handler(device_id)
            print(f"[Thread-{thread_id}] Scanning band {band_start/1e6:.3f}-{band_end/1e6:.3f} MHz with device {device_id}")
            
            # Scan frequencies in the band
            current_freq = band_start
            while current_freq <= band_end:
                run_features = []
                
                # Multiple runs per frequency for better reliability
                for _ in range(self.config.runs_per_freq):
                    iq_samples = self._read_samples(sdr_handler, current_freq)
                    
                    if iq_samples is not None:
                        features = FeatureExtractor.extract_features(iq_samples)
                        run_features.append(features)
                
                # Process and save features if we have data
                if run_features:
                    # Average features across runs
                    avg_features = np.mean(run_features, axis=0)
                    
                    # Apply PCA if available
                    if self.pca:
                        with self.lock:  # Thread-safe PCA transformation
                            reduced_features = self.pca.transform([avg_features])[0]
                        # Save the frequency and reduced features
                        data = [current_freq] + reduced_features.tolist()
                    else:
                        # Save raw features if PCA not available
                        data = [current_freq] + avg_features.tolist()
                    
                    # Save to CSV
                    self.csv_writer.write_row(data)
                    print(f"[Thread-{thread_id}] Saved data for {current_freq / 1e6:.3f} MHz")
                
                # Move to next frequency
                current_freq += self.config.freq_step
                
        except Exception as e:
            print(f"[Thread-{thread_id}] Error scanning band {band_start / 1e6:.3f}-{band_end / 1e6:.3f} MHz: {e}")
            
        finally:
            if sdr_handler:
                sdr_handler.close()
    
    def scan_bands_parallel(self):
        """
        Scan all bands in parallel using available devices.
        """
        # Find available SDR devices and get device IDs
        device_ids = self._get_device_ids()
        device_count = len(device_ids)
        
        if device_count == 0:
            raise RuntimeError(f"No available SDR devices of type {self.config.sdr_type}")
            
        # Initialize PCA
        self.initialize_pca()
        
        # Determine number of threads
        num_threads = min(device_count, len(self.config.ham_bands))
        print(f"Using {num_threads} threads (min of {device_count} devices and {len(self.config.ham_bands)} bands)")
        
        # Launch parallel scanning threads
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            
            for i, (band_start, band_end) in enumerate(self.config.ham_bands):
                device_id = device_ids[i % device_count]  # Assign device in round-robin fashion
                print(f"Assigning band {band_start / 1e6:.3f}-{band_end / 1e6:.3f} MHz to device {device_id}")
                
                # Submit task to thread pool
                futures.append(
                    executor.submit(self.scan_band, device_id, band_start, band_end)
                )
            
            # Wait for all threads to complete
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in scanning thread: {e}")
    
    def _create_sdr_handler(self, device_id: str) -> SDRHandler:
        """
        Create and initialize an SDR handler for a specific device.
        
        Args:
            device_id: Device identifier
            
        Returns:
            Initialized SDR handler
        """
        # Create custom SDR handler for threading
        sdr_handler = ThreadedSDRHandler(
            self.config.sdr_type,
            self.config.gain_value,
            self.config.sample_rate,
            device_id
        )
        
        # Initialize SDR
        sdr_handler.open()
        sdr_handler.setup_stream()
        
        return sdr_handler
    
    def _read_samples(self, sdr_handler: SDRHandler, frequency: float) -> Optional[np.ndarray]:
        """
        Read samples from an SDR at a specific frequency.
        
        Args:
            sdr_handler: SDR handler to use
            frequency: Frequency to sample
            
        Returns:
            IQ samples or None if read failed
        """
        try:
            sdr_handler.set_frequency(frequency)
            return sdr_handler.read_samples()
        except Exception as e:
            thread_id = threading.get_ident()
            print(f"[Thread-{thread_id}] Error reading samples at {frequency / 1e6:.3f} MHz: {e}")
            return None
    
    def _get_device_ids(self) -> List[str]:
        """
        Get available device IDs.
        
        Returns:
            List of device IDs
        """
        import SoapySDR
        from SoapySDR import Device as SoapyDevice
        
        # Enumerate available SDR devices
        device_dicts = [dict(dev) for dev in SoapyDevice.enumerate()]
        device_list = [dev for dev in device_dicts if dev['driver'] == self.config.sdr_type]
        device_count = len(device_list)
        
        print(f"Found {device_count} devices of type {self.config.sdr_type}")
        for i, dev in enumerate(device_list):
            print(f"Device {i}: {dev}")
        
        # Create device IDs list
        if self.config.sdr_type == "rtlsdr":
            return [dev['serial'] for dev in device_list]  # Use serials for RTL-SDR
        else:
            return [str(i) for i in range(device_count)]  # Use indexes for other SDRs


class ThreadedSDRHandler(SDRHandler):
    """Thread-safe SDR handler with device ID tracking."""
    
    def __init__(self, sdr_type: str, gain_value: float, sample_rate: float, device_id: str):
        """
        Initialize the threaded SDR handler.
        
        Args:
            sdr_type: Type of SDR device
            gain_value: Gain value to set
            sample_rate: Sample rate in Hz
            device_id: Device identifier (serial for RTL-SDR, index for others)
        """
        super().__init__(sdr_type, gain_value, sample_rate)
        self.device_id = device_id
        self.thread_id = threading.get_ident()
    
    def open(self):
        """
        Override open method to connect to specific device.
        """
        import SoapySDR
        
        print(f"[Thread-{self.thread_id}] Opening SDR Device {self.device_id}")
        
        try:
            # Connect to appropriate device based on type
            if self.sdr_type == "rtlsdr":
                self.sdr = SoapySDR.Device(dict(driver=self.sdr_type, serial=self.device_id))
                print(f"[Thread-{self.thread_id}] Setting gain to {self.gain_value}")
                self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.gain_value)
            else:
                self.sdr = SoapySDR.Device(dict(driver=self.sdr_type, index=self.device_id))
                try:
                    gain_names = self.sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0)
                    for name in gain_names:
                        self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, name, self.gain_value)
                except Exception as e:
                    print(f"[Thread-{self.thread_id}] Warning: Failed to set gain: {e}")
                    
            print(f"[Thread-{self.thread_id}] Opened SDR Device {self.device_id}")
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.sample_rate)
            print(f"[Thread-{self.thread_id}] Sample Rate set to {self.sample_rate}")
            
        except Exception as e:
            print(f"[Thread-{self.thread_id}] Error opening SDR device: {e}")
            raise RuntimeError(f"Failed to initialize SDR device {self.device_id}: {e}")
    
    def setup_stream(self):
        """Override setup_stream method to add thread ID to logs."""
        if not self.sdr:
            raise RuntimeError("SDR device not initialized. Call open() first.")
            
        import SoapySDR
        self.stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self.sdr.activateStream(self.stream)
        print(f"[Thread-{self.thread_id}] Stream setup complete")
    
    def read_samples(self, sample_size: int = 256 * 1024) -> Optional[np.ndarray]:
        """Override read_samples method to add thread ID to logs."""
        if not self.sdr or not self.stream:
            raise RuntimeError("SDR stream not initialized. Call setup_stream() first.")
            
        buff = np.zeros(sample_size, dtype=np.complex64)
        sr = self.sdr.readStream(self.stream, [buff], len(buff))
        print(f"[Thread-{self.thread_id}] Stream Read: {sr.ret if hasattr(sr, 'ret') else 'unknown'} samples")
        
        if hasattr(sr, 'ret') and sr.ret > 0:  # Only process valid samples
            return buff[:sr.ret]  # Extract only valid IQ samples
        return None
    
    def close(self):
        """Override close method to add thread ID to logs."""
        if self.stream is not None:
            try:
                import SoapySDR
                self.sdr.deactivateStream(self.stream)
                self.sdr.closeStream(self.stream)
                self.stream = None
                print(f"[Thread-{self.thread_id}] Stream Closed")
            except Exception as e:
                print(f"[Thread-{self.thread_id}] Error closing stream: {e}")
                
        if self.sdr is not None:
            try:
                self.sdr.close()
                self.sdr = None
                print(f"[Thread-{self.thread_id}] Closed SDR Device {self.device_id}")
            except Exception as e:
                print(f"[Thread-{self.thread_id}] Error closing SDR: {e}")


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        Parsed arguments object
    """
    parser = argparse.ArgumentParser(description="Multi-threaded Spectrum Data Gathering with SDR.")
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file (default: config.ini)")
    parser.add_argument("-d", "--duration", type=float, default=10,
                        help="Duration in minutes (default: 10)")
    parser.add_argument("-o", "--output", type=str, default="refactored_collected_iq_data.csv",
                        help="Path to the output file (default: refactored_collected_iq_data.csv)")

    return parser.parse_args()


def main():
    """Main entry point for the application."""
    try:
        # Parse command line arguments
        args = parse_arguments()

        # Display configuration
        print(f"Using config file: {args.config}")
        print(f"Monitoring duration: {args.duration} minutes")
        print(f"Saving to: {args.output}")

        # Read configuration
        config = ConfigReader.read_config(args.config)
        print(f"Configured for {len(config.ham_bands)} frequency bands")
        
        # Create thread-safe CSV writer
        # Using feature extractor to get the feature count
        sample_features = FeatureExtractor.extract_features(np.zeros(10, dtype=np.complex64))
        csv_writer = ThreadSafeCSVWriter(args.output, len(sample_features))
        
        # Create band scanner
        scanner = BandScanner(config, csv_writer)
        
        # Start parallel scanning
        print(f"Starting IQ data collection for {args.duration} minutes...")
        scanner.scan_bands_parallel()
        print("Data collection completed successfully.")

    except KeyboardInterrupt:
        print("Data collection interrupted by user.")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
