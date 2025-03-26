#!/usr/bin/env python3
"""
Device identification utilities for SpectrumAlert.
"""

import os
import re
import subprocess


class DeviceIdentifier:
    """Handles device identification functions."""
    
    @staticmethod
    def get_primary_mac() -> str:
        """
        Retrieves the primary MAC address (uppercase, no colons).
        
        Returns:
            MAC address string or "UNKNOWNMAC" if retrieval fails
        """
        try:
            # Get MAC address using ip link (Linux)
            cmd = "ip link show | grep -m 1 'link/ether' | awk '{print $2}'"
            mac_output = subprocess.check_output(cmd, shell=True, text=True).strip()
            
            # Remove colons and convert to uppercase
            return re.sub(r'[:]', '', mac_output).upper()
        except Exception as e:
            print(f"❌ Error getting MAC address: {e}")
            return "UNKNOWNMAC"
    
    @staticmethod
    def get_device_name() -> str:
        """
        Retrieves the device name from environment variable or generates it.
        
        Returns:
            Device name string
        """
        # Check if running on Balena
        device_name = os.getenv("BALENA_DEVICE_NAME_AT_INIT")

        if not device_name:
            try:
                host = subprocess.check_output("hostname", shell=True, text=True).strip()
                mac = DeviceIdentifier.get_primary_mac()
                device_name = f"{host}-{mac}"
            except Exception as e:
                print(f"❌ Error getting fallback device name: {e}")
                device_name = "unknown-device"

        return device_name