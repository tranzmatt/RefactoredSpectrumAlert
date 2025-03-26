#!/usr/bin/env python3

"""
Example showing how to use a specific SDR device by serial number.
"""

import argparse
import sys

from spectrum_utils import (
    ConfigReader, SDRHandler, FeatureExtractor, format_frequency
)


def list_available_devices():
    """List all available SDR devices."""
    print("Listing all available SDR devices:")
    
    # Get all devices
    all_devices = SDRHandler.list_available_devices()
    
    if not all_devices:
        print("No SDR devices found.")
        return
        
    # Group by device type
    device_types = {}
    for dev in all_devices:
        driver = dev.get('driver', 'unknown')
        if driver not in device_types:
            device_types[driver] = []
        device_types[driver].append(dev)
        
    # Print device information
    for driver, devices in device_types.items():
        print(f"\n{driver.upper()} Devices ({len(devices)} found):")
        
        for i, dev in enumerate(devices):
            print(f"  Device {i}:")
            
            # Print key device details
            if 'label' in dev:
                print(f"    Label: {dev['label']}")
            if 'serial' in dev:
                print(f"    Serial: {dev['serial']}")
            if 'device' in dev:
                print(f"    Device: {dev['device']}")
                
            # Print other info
            for key, value in dev.items():
                if key not in ('label', 'serial', 'device', 'driver'):
                    print(f"    {key}: {value}")
            
            print()


def scan_with_specific_device(serial_number, sdr_type):
    """
    Scan frequencies using a specific device by serial number.
    
    Args:
        serial_number: Serial number of the device to use
        sdr_type: Type of SDR (rtlsdr, hackrf, etc.)
    """
    print(f"Scanning with {sdr_type} device {serial_number}")
    
    # Create SDR handler with specific serial
    sdr_handler = SDRHandler(
        sdr_type=sdr_type,
        gain_value=20.0,
        device_serial=serial_number
    )
    
    try:
        # Open the device
        sdr_handler.open()
        
        # Setup stream
        sdr_handler.setup_stream()
        
        # Scan some example frequencies
        frequencies = [462e6, 467e6, 152.5e6]
        
        for freq in frequencies:
            formatted_freq = format_frequency(freq)
            print(f"\nScanning frequency {formatted_freq}:")
            
            # Set frequency
            sdr_handler.set_frequency(freq)
            
            # Read samples
            iq_samples = sdr_handler.read_samples()
            
            if iq_samples is not None:
                # Calculate signal strength
                signal_strength = FeatureExtractor.calculate_signal_strength(iq_samples)
                print(f"  Signal strength: {signal_strength:.2f} dB")
                
                # Extract features
                features = FeatureExtractor.extract_features(iq_samples)
                print(f"  Features: {features[:2]}")  # Show first two features
            else:
                print("  No samples received")
    
    finally:
        # Always clean up
        sdr_handler.close()


def main():
    """Main entry point for the example."""
    parser = argparse.ArgumentParser(description="Example of using specific SDR devices.")
    parser.add_argument("--list", action="store_true", help="List all available SDR devices")
    parser.add_argument("--serial", type=str, help="Specify a device by serial number")
    parser.add_argument("--type", type=str, default="hackrf", help="SDR type (default: hackrf)")
    
    args = parser.parse_args()
    
    if args.list:
        # List available devices
        list_available_devices()
    elif args.serial:
        # Use specific device
        scan_with_specific_device(args.serial, args.type)
    else:
        print("Please specify either --list to list devices or --serial to use a specific device.")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
