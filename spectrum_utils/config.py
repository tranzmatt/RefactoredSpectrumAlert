#!/usr/bin/env python3
"""
Configuration classes for SpectrumAlert components.
"""

import configparser
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

# Default constants
DEFAULT_SAMPLE_SIZE = 128 * 1024
DEFAULT_SAMPLE_RATE = 2.048e6
DEFAULT_FREQ_STEP = 500e3
DEFAULT_RUNS_PER_FREQ = 5
DEFAULT_GAIN = 20.0


@dataclass
class Config:
    """Configuration container for spectrum operations."""
    ham_bands: List[Tuple[float, float]]
    freq_step: float
    sample_rate: float
    runs_per_freq: int
    sdr_type: str
    min_db: float
    gain_value: float
    device_serial: Optional[str] = None  # Optional device serial parameter


class ConfigReader:
    """Handles reading and parsing of configuration files."""
    
    @staticmethod
    def read_config(config_file='config.ini') -> Config:
        """
        Read and parse the configuration file.
        
        Args:
            config_file: Path to the configuration file
            
        Returns:
            Config object with parsed configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            ValueError: If required config sections or values are missing
        """
        config = configparser.ConfigParser()

        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Config file '{config_file}' not found.")

        config.read(config_file)
        
        # Validate that required sections exist
        if 'HAM_BANDS' not in config:
            raise ValueError("'HAM_BANDS' section missing in the config file.")
        if 'GENERAL' not in config:
            raise ValueError("'GENERAL' section missing in the config file.")

        # Parse HAM bands
        ham_bands_str = config['HAM_BANDS'].get('bands')
        if not ham_bands_str:
            raise ValueError("Missing 'bands' entry in 'HAM_BANDS' section")
            
        ham_bands = []
        for band in ham_bands_str.split(','):
            try:
                start, end = band.split('-')
                ham_bands.append((float(start), float(end)))
            except ValueError:
                raise ValueError(f"Invalid frequency range format: {band}. Expected 'start-end'.")

        # Parse general settings with defaults
        general = config['GENERAL']
        freq_step = float(general.get('freq_step', str(DEFAULT_FREQ_STEP)))
        sample_rate = float(general.get('sample_rate', str(DEFAULT_SAMPLE_RATE)))
        runs_per_freq = int(general.get('runs_per_freq', str(DEFAULT_RUNS_PER_FREQ)))
        sdr_type = general.get('sdr_type', 'rtlsdr')
        min_db = float(general.get('min_db', '-80.0'))
        gain_value = float(general.get('gain_value', str(DEFAULT_GAIN)))
        
        # Get optional device serial
        device_serial = general.get('device_serial', None)

        return Config(
            ham_bands=ham_bands,
            freq_step=freq_step,
            sample_rate=sample_rate,
            runs_per_freq=runs_per_freq,
            sdr_type=sdr_type,
            min_db=min_db,
            gain_value=gain_value,
            device_serial=device_serial
        )
