#!/usr/bin/env python3
"""
General utilities for SpectrumAlert.
"""

from typing import Any


def safe_float_conversion(value: Any, default: float = 0.0) -> float:
    """
    Safely convert a value to float.
    
    Args:
        value: Value to convert
        default: Default value to return if conversion fails
        
    Returns:
        Converted float or default value
    """
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def format_frequency(freq_hz: float, unit: str = 'auto') -> str:
    """
    Format frequency in human-readable form.
    
    Args:
        freq_hz: Frequency in Hz
        unit: Unit to use ('Hz', 'kHz', 'MHz', 'GHz', or 'auto')
        
    Returns:
        Formatted frequency string
    """
    if unit == 'auto':
        if freq_hz < 1e3:
            unit = 'Hz'
        elif freq_hz < 1e6:
            unit = 'kHz'
        elif freq_hz < 1e9:
            unit = 'MHz'
        else:
            unit = 'GHz'
    
    if unit == 'Hz':
        return f"{freq_hz:.0f} Hz"
    elif unit == 'kHz':
        return f"{freq_hz/1e3:.3f} kHz"
    elif unit == 'MHz':
        return f"{freq_hz/1e6:.3f} MHz"
    elif unit == 'GHz':
        return f"{freq_hz/1e9:.3f} GHz"
    else:
        return f"{freq_hz} Hz"