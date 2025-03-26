#!/usr/bin/env python3
"""
Setup script for SpectrumAlert utilities.
"""

from setuptools import setup, find_packages

setup(
    name="spectrum_utils",
    version="0.1.0",
    description="Common utilities for SpectrumAlert components",
    author="SpectrumAlert Team",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "gpsd-py3>=0.3.0",
        "paho-mqtt>=2.0.0",
    ],
    python_requires=">=3.7",
)
