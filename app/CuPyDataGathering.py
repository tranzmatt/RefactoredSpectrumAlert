#!/usr/bin/env python3
"""
SpectrumAlert DataGathering - Multi-threaded SDR-based spectrum data collection (GPU-accelerated).
"""

import argparse
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple, Optional, Any

import cupy as np
import numpy as cpu_np
import SoapySDR
from sklearn.decomposition import PCA

from spectrum_utils import (
    Config, ConfigReader, SDRHandler, FeatureExtractor, CSVDataHandler
)

# Thread-safe CSV file writer
class ThreadSafeCSVWriter:
    def __init__(self, filename: str, feature_count: int):
        self.lock = threading.Lock()
        self.csv_handler = CSVDataHandler(
            filename, 
            header=self._get_header_columns(feature_count)
        )

    def write_row(self, data: List[Any]):
        with self.lock:
            self.csv_handler.write_row(data)

    def _get_header_columns(self, feature_count: int) -> List[str]:
        header = ['Frequency']
        feature_names = [
            'Mean_Amplitude', 'Std_Amplitude', 'Mean_FFT_Magnitude', 
            'Std_FFT_Magnitude', 'Skew_Amplitude', 'Kurt_Amplitude', 
            'Skew_Phase', 'Kurt_Phase', 'Cyclo_Autocorr',
            'Spectral_Entropy', 'PAPR', 'Band_Energy_Ratio'
        ]
        while len(feature_names) < feature_count:
            feature_names.append(f'Feature_{len(feature_names) + 1}')
        return header + feature_names[:feature_count]


class BandScanner:
    def __init__(self, config: Config, csv_writer: ThreadSafeCSVWriter):
        self.config = config
        self.csv_writer = csv_writer
        self.pca = None
        self.lock = threading.Lock()

    def initialize_pca(self):
        print("Collecting PCA training data...")
        pca_training_data = []
        sdr_handler = None

        try:
            sdr_handler = self._create_sdr_handler("0")
            for band_start, band_end in self.config.ham_bands:
                iq_samples = self._read_samples(sdr_handler, band_start)
                if iq_samples is not None:
                    features = FeatureExtractor.extract_features(iq_samples)
                    pca_training_data.append(features)
                    print(f"Collected PCA training data at {band_start / 1e6:.3f} MHz")

            if not pca_training_data:
                raise RuntimeError("Failed to collect any PCA training data")

            num_features = len(pca_training_data[0])
            n_components = min(8, len(pca_training_data), num_features)
            self.pca = PCA(n_components=n_components)
            self.pca.fit(cpu_np.asarray(pca_training_data))
            print(f"PCA model trained with {len(pca_training_data)} samples, {n_components} components")
            return self.pca

        finally:
            if sdr_handler:
                sdr_handler.close()

    def scan_band(self, device_id: str, band_start: float, band_end: float):
        sdr_handler = None
        thread_id = threading.get_ident()
        try:
            sdr_handler = self._create_sdr_handler(device_id)
            print(f"[Thread-{thread_id}] Scanning band {band_start/1e6:.3f}-{band_end/1e6:.3f} MHz with device {device_id}")
            current_freq = band_start
            while current_freq <= band_end:
                run_features = []
                for _ in range(self.config.runs_per_freq):
                    iq_samples = self._read_samples(sdr_handler, current_freq)
                    if iq_samples is not None:
                        features = FeatureExtractor.extract_features(iq_samples)
                        run_features.append(features)

                if run_features:
                    avg_features = np.mean(np.asarray(run_features), axis=0).get()
                    if self.pca:
                        with self.lock:
                            reduced_features = self.pca.transform([avg_features])[0]
                        data = [current_freq] + reduced_features.tolist()
                    else:
                        data = [current_freq] + avg_features.tolist()
                    self.csv_writer.write_row(data)
                    print(f"[Thread-{thread_id}] Saved data for {current_freq / 1e6:.3f} MHz")
                current_freq += self.config.freq_step
        except Exception as e:
            print(f"[Thread-{thread_id}] Error scanning band {band_start / 1e6:.3f}-{band_end / 1e6:.3f} MHz: {e}")
        finally:
            if sdr_handler:
                sdr_handler.close()

    def scan_bands_parallel(self):
        device_ids = self._get_device_ids()
        if not device_ids:
            raise RuntimeError(f"No available SDR devices of type {self.config.sdr_type}")

        self.initialize_pca()
        num_threads = min(len(device_ids), len(self.config.ham_bands))
        print(f"Using {num_threads} threads")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = []
            for i, (band_start, band_end) in enumerate(self.config.ham_bands):
                device_id = device_ids[i % len(device_ids)]
                futures.append(executor.submit(self.scan_band, device_id, band_start, band_end))

            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in scanning thread: {e}")

    def _create_sdr_handler(self, device_id: str) -> SDRHandler:
        handler = ThreadedSDRHandler(
            self.config.sdr_type,
            self.config.gain_value,
            self.config.sample_rate,
            device_id
        )
        handler.open()
        handler.setup_stream()
        return handler

    def _read_samples(self, sdr_handler: SDRHandler, frequency: float) -> Optional[np.ndarray]:
        try:
            sdr_handler.set_frequency(frequency)
            return sdr_handler.read_samples()
        except Exception as e:
            thread_id = threading.get_ident()
            print(f"[Thread-{thread_id}] Error reading samples at {frequency / 1e6:.3f} MHz: {e}")
            return None

    def _get_device_ids(self) -> List[str]:
        from SoapySDR import Device as SoapyDevice
        device_dicts = [dict(dev) for dev in SoapyDevice.enumerate()]
        device_list = [dev for dev in device_dicts if dev['driver'] == self.config.sdr_type]
        print(f"Found {len(device_list)} devices of type {self.config.sdr_type}")
        for i, dev in enumerate(device_list):
            print(f"Device {i}: {dev}")
        if self.config.sdr_type == "rtlsdr":
            return [dev['serial'] for dev in device_list]
        return [str(i) for i in range(len(device_list))]


class ThreadedSDRHandler(SDRHandler):
    def __init__(self, sdr_type: str, gain_value: float, sample_rate: float, device_id: str):
        super().__init__(sdr_type, gain_value, sample_rate)
        self.device_id = device_id
        self.thread_id = threading.get_ident()

    def open(self):
        print(f"[Thread-{self.thread_id}] Opening SDR Device {self.device_id}")
        try:
            if self.sdr_type == "rtlsdr":
                self.sdr = SoapySDR.Device(dict(driver=self.sdr_type, serial=self.device_id))
                self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.gain_value)
            else:
                self.sdr = SoapySDR.Device(dict(driver=self.sdr_type, index=self.device_id))
                for name in self.sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0):
                    self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, name, self.gain_value)
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.sample_rate)
            print(f"[Thread-{self.thread_id}] SDR Device {self.device_id} opened")
        except Exception as e:
            print(f"[Thread-{self.thread_id}] Error opening SDR device: {e}")
            raise RuntimeError(f"Failed to initialize SDR device {self.device_id}: {e}")

    def setup_stream(self):
        self.stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self.sdr.activateStream(self.stream)
        print(f"[Thread-{self.thread_id}] Stream setup complete")

    def read_samples(self, sample_size: int = 256 * 1024) -> Optional[np.ndarray]:
        buff = np.zeros(sample_size, dtype=np.complex64)
        sr = self.sdr.readStream(self.stream, [buff], len(buff))
        print(f"[Thread-{self.thread_id}] Stream Read: {sr.ret if hasattr(sr, 'ret') else 'unknown'} samples")
        if hasattr(sr, 'ret') and sr.ret > 0:
            return buff[:sr.ret]
        return None

    def close(self):
        if self.stream is not None:
            try:
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
    parser = argparse.ArgumentParser(description="Multi-threaded Spectrum Data Gathering with SDR.")
    parser.add_argument("-c", "--config", type=str, default="config.ini",
                        help="Path to the configuration file (default: config.ini)")
    parser.add_argument("-d", "--duration", type=float, default=10,
                        help="Duration in minutes (default: 10)")
    parser.add_argument("-o", "--output", type=str, default="refactored_collected_iq_data.csv",
                        help="Path to the output file (default: refactored_collected_iq_data.csv)")
    return parser.parse_args()


def main():
    try:
        args = parse_arguments()
        print(f"Using config file: {args.config}")
        print(f"Monitoring duration: {args.duration} minutes")
        print(f"Saving to: {args.output}")

        config = ConfigReader.read_config(args.config)
        print(f"Configured for {len(config.ham_bands)} frequency bands")

        sample_features = FeatureExtractor.extract_features(np.zeros(10, dtype=np.complex64))
        csv_writer = ThreadSafeCSVWriter(args.output, len(sample_features))

        scanner = BandScanner(config, csv_writer)
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

