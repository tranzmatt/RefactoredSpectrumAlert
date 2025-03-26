#!/usr/bin/env python3
"""
GPS coordinate handling for SpectrumAlert.
"""

import os
from typing import Tuple, Optional

import gpsd


class GPSHandler:
    """Handles GPS coordinate retrieval."""
    
    @staticmethod
    def get_coordinates() -> Tuple[Optional[float], Optional[float], Optional[float]]:
        """
        Retrieves GPS coordinates from gpsd or fixed values from environment variables.
        
        Returns:
            Tuple of (latitude, longitude, altitude) or (None, None, None) if unavailable
        """
        gps_source = os.getenv("GPS_SOURCE", "fixed").lower()

        if gps_source == "fixed":
            # Get fixed coordinates from environment variables
            altitude = float(os.getenv("GPS_FIX_ALT", 1))
            latitude = float(os.getenv("GPS_FIX_LAT", 0))
            longitude = float(os.getenv("GPS_FIX_LON", 0))
            print(f"Using fixed GPS coordinates: {latitude}, {longitude}, {altitude}")
            return latitude, longitude, altitude

        if gps_source == "gpsd":
            try:
                # Connect to gpsd
                gpsd.connect(host="localhost", port=2947)
                
                # Get GPS data
                gps_data = gpsd.get_current()

                if gps_data is None:
                    print("⚠️ No GPS data available. GPS may not be active.")
                    return None, None, None

                if gps_data.mode >= 2:  # 2D or 3D fix
                    latitude = gps_data.lat
                    longitude = gps_data.lon
                    # Altitude available only in 3D mode
                    altitude = gps_data.alt if gps_data.mode == 3 else None
                    print(f"📍 GPSD Coordinates: {latitude}, {longitude}, Alt: {altitude}m")
                    return latitude, longitude, altitude
                else:
                    print("⚠️ No GPS fix yet.")
            except Exception as e:
                print(f"❌ GPSD Error: {e}")
        else:
            print(f"Unsupported GPS source: {gps_source}")

        return None, None, None