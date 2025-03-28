#!/usr/bin/env python3
"""
Multi-SDR handler utilities for SpectrumAlert.
"""

import threading
import time
from typing import Dict, List, Tuple, Optional, Any, Union

from .sdr_handler import SDRHandler


class SDRAssignment:
    """Container for SDR device and its assigned frequency bands."""
    
    def __init__(self, 
                 sdr_handler: SDRHandler, 
                 freq_ranges: List[Tuple[float, float]], 
                 device_id: str,
                 antenna_info: str = "Unknown"):
        """
        Initialize an SDR assignment.
        
        Args:
            sdr_handler: The initialized SDR handler object
            freq_ranges: List of frequency ranges (tuples of start_freq, end_freq in Hz)
            device_id: Unique identifier for this SDR
            antenna_info: Description of the antenna connected to this SDR
        """
        self.sdr_handler = sdr_handler
        self.freq_ranges = freq_ranges
        self.device_id = device_id
        self.antenna_info = antenna_info
        self.is_active = False
        
    def __str__(self) -> str:
        """String representation of the SDR assignment."""
        ranges_str = ", ".join([f"{start/1e6:.1f}-{end/1e6:.1f}MHz" for start, end in self.freq_ranges])
        return f"SDR {self.device_id} ({self.sdr_handler.sdr_type}): {ranges_str} - {self.antenna_info}"


class MultiSDRHandler:
    """Handles multiple SDR devices with their frequency assignments."""
    
    def __init__(self):
        """Initialize the multi-SDR handler."""
        self.sdr_assignments = {}
        self.threads = []
        self.stop_event = threading.Event()
        
    def add_sdr(self, 
                device_id: str,
                sdr_type: str, 
                freq_ranges: List[Tuple[float, float]],
                gain_value: float = 20.0,
                device_serial: Optional[str] = None,
                antenna_info: str = "Unknown") -> str:
        """
        Add an SDR device with its assigned frequency ranges.
        
        Args:
            device_id: Unique identifier for this SDR
            sdr_type: Type of SDR (e.g., 'rtlsdr', 'hackrf')
            freq_ranges: List of frequency ranges to scan with this SDR
            gain_value: Gain setting for the SDR
            device_serial: Optional serial number for the device
            antenna_info: Description of the connected antenna
            
        Returns:
            The device ID
        """
        # Create SDR handler
        sdr_handler = SDRHandler(
            sdr_type=sdr_type,
            gain_value=gain_value,
            device_serial=device_serial
        )
        
        # Create assignment
        assignment = SDRAssignment(
            sdr_handler=sdr_handler,
            freq_ranges=freq_ranges,
            device_id=device_id,
            antenna_info=antenna_info
        )
        
        # Store assignment
        self.sdr_assignments[device_id] = assignment
        
        return device_id
    
    def load_from_config(self, config_file: str) -> None:
        """
        Load SDR assignments from a configuration file.
        
        Args:
            config_file: Path to the configuration file
        """
        import configparser
        config = configparser.ConfigParser()
        config.read(config_file)
        
        # Process each SDR section in the config
        for section in config.sections():
            if section.startswith('SDR_'):
                device_id = section
                
                # Get SDR parameters
                sdr_type = config[section].get('type', 'rtlsdr')
                serial = config[section].get('serial', None)
                gain = float(config[section].get('gain', '20.0'))
                
                # Get frequency ranges this SDR is optimized for
                freq_ranges_str = config[section].get('freq_ranges', '')
                freq_ranges = []
                
                for range_str in freq_ranges_str.split(','):
                    range_str = range_str.strip()
                    if range_str and '-' in range_str:
                        start, end = range_str.split('-')
                        freq_ranges.append((float(start), float(end)))
                
                # Add SDR to handler
                self.add_sdr(
                    device_id=device_id,
                    sdr_type=sdr_type,
                    freq_ranges=freq_ranges,
                    gain_value=gain,
                    device_serial=serial,
                    antenna_info=config[section].get('antenna_info', 'Unknown')
                )
    
    def initialize_all(self) -> None:
        """Initialize all SDR devices."""
        for device_id, assignment in self.sdr_assignments.items():
            try:
                # Open and set up the SDR
                assignment.sdr_handler.open()
                assignment.sdr_handler.setup_stream()
                assignment.is_active = True
                print(f"Initialized {assignment}")
            except Exception as e:
                print(f"Error initializing SDR {device_id}: {e}")
                assignment.is_active = False
    
    def close_all(self) -> None:
        """Close all SDR devices."""
        for device_id, assignment in self.sdr_assignments.items():
            if assignment.is_active:
                try:
                    assignment.sdr_handler.close()
                    assignment.is_active = False
                    print(f"Closed SDR {device_id}")
                except Exception as e:
                    print(f"Error closing SDR {device_id}: {e}")
    
    def start_data_gathering(self, 
                             gather_function: callable, 
                             duration_minutes: float,
                             output_prefix: str = "collected_data",
                             **kwargs) -> None:
        """
        Start data gathering with all active SDRs.
        
        Args:
            gather_function: Function to call for each SDR (must accept sdr_handler, freq_ranges, output_file)
            duration_minutes: Duration to gather data in minutes
            output_prefix: Prefix for output files
            **kwargs: Additional arguments to pass to the gather function
        """
        self.stop_event.clear()
        end_time = time.time() + (duration_minutes * 60)
        
        # Start a thread for each active SDR
        self.threads = []
        for device_id, assignment in self.sdr_assignments.items():
            if assignment.is_active:
                output_file = f"{output_prefix}_{device_id}.csv"
                
                thread = threading.Thread(
                    target=self._gathering_thread,
                    args=(
                        gather_function,
                        assignment,
                        output_file,
                        end_time,
                        kwargs
                    ),
                    name=f"SDR-{device_id}"
                )
                
                self.threads.append(thread)
                thread.start()
                print(f"Started gathering thread for SDR {device_id}")
        
        # Wait for all threads to complete or timeout
        for thread in self.threads:
            thread.join()
    
    def stop_data_gathering(self) -> None:
        """Stop all data gathering threads."""
        self.stop_event.set()
        
        # Wait for all threads to complete
        for thread in self.threads:
            thread.join(timeout=10)  # Wait up to 10 seconds
            
        self.threads = []
    
    def _gathering_thread(self, 
                         gather_function: callable, 
                         assignment: SDRAssignment, 
                         output_file: str,
                         end_time: float,
                         kwargs: Dict) -> None:
        """
        Thread function for data gathering.
        
        Args:
            gather_function: Function to call for data gathering
            assignment: SDR assignment to use
            output_file: Output file path
            end_time: Time to stop gathering
            kwargs: Additional arguments
        """
        try:
            # Call the gather function with a stop condition
            gather_function(
                sdr_handler=assignment.sdr_handler,
                freq_ranges=assignment.freq_ranges,
                output_file=output_file,
                stop_event=self.stop_event,
                end_time=end_time,
                device_id=assignment.device_id,
                **kwargs
            )
            
        except Exception as e:
            print(f"Error in gathering thread for SDR {assignment.device_id}: {e}")
            import traceback
            traceback.print_exc()
