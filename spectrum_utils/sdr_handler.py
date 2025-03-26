#!/usr/bin/env python3
"""
SDR device handling for SpectrumAlert.
"""

from typing import List, Optional

import numpy as np
import SoapySDR
from SoapySDR import Device as SoapyDevice

# Suppress INFO messages
SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_WARNING)

# Default constants
DEFAULT_SAMPLE_SIZE = 128 * 1024
DEFAULT_SAMPLE_RATE = 2.048e6
DEFAULT_GAIN = 20.0


class SDRHandler:
    """Handles SDR device setup, connection and data reading."""
    
    def __init__(self, sdr_type: str, gain_value: float, sample_rate: float = DEFAULT_SAMPLE_RATE, device_serial: str = None):
        """
        Initialize the SDR handler.
        
        Args:
            sdr_type: Type of SDR device (e.g., 'rtlsdr', 'hackrf')
            gain_value: Gain value to set
            sample_rate: Sample rate in Hz
            device_serial: Optional serial number to select a specific device
            
        Raises:
            RuntimeError: If no matching SDR devices are found
        """
        self.sdr_type = sdr_type
        self.gain_value = gain_value
        self.sample_rate = sample_rate
        self.device_serial = device_serial
        self.sdr = None
        self.stream = None
        
    def open(self):
        """
        Find and connect to an SDR device.
        
        Raises:
            RuntimeError: If no devices are found or connection fails
        """
        # Enumerate available SDR devices
        device_dicts = [dict(dev) for dev in SoapyDevice.enumerate()]
        device_list = [dev for dev in device_dicts if dev['driver'] == self.sdr_type]
        device_count = len(device_list)

        if device_count == 0:
            raise RuntimeError(f"No available SDR devices of type {self.sdr_type}")
        
        print(f"Found {device_count} devices of type {self.sdr_type}")
        for i, dev in enumerate(device_list):
            print(f"Device {i}: {dev}")
            
        # Connect to device by serial or index
        if self.device_serial:
            self._connect_by_serial(device_list)
        else:
            self._connect_by_index(device_count)
            
        # Set sample rate
        if self.sdr:
            self.sdr.setSampleRate(SoapySDR.SOAPY_SDR_RX, 0, self.sample_rate)
            print(f"Sample rate set to {self.sample_rate}")
            
    def _connect_by_serial(self, device_list):
        """
        Connect to an SDR device by serial number.
        
        Args:
            device_list: List of available devices
            
        Raises:
            RuntimeError: If the device with specified serial is not found
        """
        # Find the device with matching serial
        matching_devices = []
        
        for dev in device_list:
            # Different SDR types may have different ways to store serial
            if 'serial' in dev and dev['serial'] == self.device_serial:
                matching_devices.append(dev)
            # HackRF serial is sometimes stored differently
            elif self.sdr_type == 'hackrf' and 'serial' in dev and self.device_serial in dev['serial']:
                matching_devices.append(dev)
        
        if not matching_devices:
            raise RuntimeError(f"No {self.sdr_type} device found with serial '{self.device_serial}'")
        
        # Use the first matching device
        dev = matching_devices[0]
        print(f"Connecting to {self.sdr_type} device with serial {self.device_serial}")
        
        # Connect based on device type
        try:
            if self.sdr_type == "rtlsdr":
                self.sdr = SoapySDR.Device(dict(driver=self.sdr_type, serial=dev['serial']))
                self._set_rtlsdr_gain()
            else:
                # For HackRF, we can use either index or serial
                if 'serial' in dev:
                    self.sdr = SoapySDR.Device(dict(driver=self.sdr_type, serial=dev['serial']))
                else:
                    # Find the index in the original list
                    idx = next(i for i, d in enumerate(device_list) if d == dev)
                    self.sdr = SoapySDR.Device(dict(driver=self.sdr_type, index=str(idx)))
                self._set_sdr_gain()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to {self.sdr_type} device with serial {self.device_serial}: {e}")
            
    def _connect_by_index(self, device_count, index=0):
        """
        Connect to an SDR device by index (defaults to first device).
        
        Args:
            device_count: Number of available devices
            index: Device index to use (default 0)
            
        Raises:
            RuntimeError: If connection fails
        """
        if index >= device_count:
            raise RuntimeError(f"Device index {index} out of range (0-{device_count-1})")
            
        print(f"Connecting to {self.sdr_type} device with index {index}")
        
        try:
            if self.sdr_type == "rtlsdr":
                self.sdr = SoapySDR.Device(dict(driver=self.sdr_type, index=str(index)))
                self._set_rtlsdr_gain()
            else:
                self.sdr = SoapySDR.Device(dict(driver=self.sdr_type, index=str(index)))
                self._set_sdr_gain()
        except Exception as e:
            raise RuntimeError(f"Failed to connect to {self.sdr_type} device with index {index}: {e}")
            
    def _set_rtlsdr_gain(self):
        """Set gain for RTL-SDR devices."""
        if self.sdr is None:
            return
            
        try:
            self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, self.gain_value)
            print(f"RTL-SDR gain set to {self.gain_value}")
        except Exception as e:
            print(f"Warning: Failed to set RTL-SDR gain: {e}")
            
    def _set_sdr_gain(self):
        """Set gain for other SDR devices with multiple gain elements."""
        if self.sdr is None:
            return
            
        try:
            gain_names = self.sdr.listGains(SoapySDR.SOAPY_SDR_RX, 0)
            for name in gain_names:
                self.sdr.setGain(SoapySDR.SOAPY_SDR_RX, 0, name, self.gain_value)
                print(f"Set {name} gain to {self.gain_value}")
        except Exception as e:
            print(f"Warning: Failed to set gain: {e}")
            
    def setup_stream(self):
        """Set up the SDR stream for reading data."""
        if not self.sdr:
            raise RuntimeError("SDR device not initialized. Call open() first.")
            
        self.stream = self.sdr.setupStream(SoapySDR.SOAPY_SDR_RX, SoapySDR.SOAPY_SDR_CF32)
        self.sdr.activateStream(self.stream)
        
    def set_frequency(self, freq: float):
        """
        Set the SDR frequency.
        
        Args:
            freq: Frequency to set in Hz
        """
        if not self.sdr:
            raise RuntimeError("SDR device not initialized. Call open() first.")
            
        self.sdr.setFrequency(SoapySDR.SOAPY_SDR_RX, 0, freq)
        
    def read_samples(self, sample_size: int = DEFAULT_SAMPLE_SIZE) -> Optional[np.ndarray]:
        """
        Read samples from the SDR.
        
        Args:
            sample_size: Number of samples to read
            
        Returns:
            Array of IQ samples or None if read failed
        """
        if not self.sdr or not self.stream:
            raise RuntimeError("SDR stream not initialized. Call setup_stream() first.")
            
        buff = np.zeros(sample_size, dtype=np.complex64)
        sr = self.sdr.readStream(self.stream, [buff], len(buff))
        
        if sr.ret > 0:  # Only process valid samples
            return buff[:sr.ret]  # Extract only valid IQ samples
        return None
        
    def close(self):
        """Clean up and close the SDR connection."""
        if self.stream is not None:
            try:
                self.sdr.deactivateStream(self.stream)
                self.sdr.closeStream(self.stream)
                self.stream = None
            except Exception as e:
                print(f"Warning: Error closing stream: {e}")
                
        if self.sdr is not None:
            try:
                self.sdr.close()
                self.sdr = None
                print("Closed SDR device.")
            except Exception as e:
                print(f"Warning: Error closing SDR: {e}")

    @staticmethod
    def list_available_devices(sdr_type=None):
        """
        List all available SDR devices or devices of a specific type.

        Args:
            sdr_type: Optional SDR type to filter (e.g., 'rtlsdr', 'hackrf')

        Returns:
            List of device dictionaries
        """
        try:
            # Use a safer approach to enumerate devices
            result = []

            # Try to enumerate all devices
            try:
                devices = SoapyDevice.enumerate()
                for dev in devices:
                    try:
                        # Convert to dict carefully
                        dev_dict = dict(dev)
                        result.append(dev_dict)
                    except Exception as e:
                        print(f"Warning: Error converting device to dict: {e}")
                        continue
            except Exception as e:
                print(f"Warning: Error enumerating devices: {e}")
                return []

            # Filter by type if specified
            if sdr_type:
                result = [dev for dev in result if dev.get('driver') == sdr_type]

            return result
        except Exception as e:
            print(f"Error listing devices: {e}")
            return []