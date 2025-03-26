#!/usr/bin/env python3
"""
CSV data handling for SpectrumAlert.
"""

import csv
import os
import threading
from typing import List, Tuple, Any


class CSVDataHandler:
    """Handles CSV file operations with thread safety."""
    
    def __init__(self, filename: str, header: List[str] = None):
        """
        Initialize the CSV data handler.
        
        Args:
            filename: Path to the CSV file
            header: Optional list of column headers
        """
        self.filename = filename
        self.header = header
        self.header_written = False
        self.lock = threading.Lock()  # Thread safety lock
        
        # Create directory if it doesn't exist
        directory = os.path.dirname(filename)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    def write_row(self, data: List[Any]):
        """
        Write a row of data to the CSV file.
        
        Args:
            data: List of values to write
        """
        # Use a lock for thread safety
        with self.lock:
            mode = 'a' if os.path.exists(self.filename) or self.header_written else 'w'
            with open(self.filename, mode, newline='') as f:
                writer = csv.writer(f)
                
                # Write header if needed
                if not self.header_written and self.header:
                    writer.writerow(self.header)
                    self.header_written = True
                
                # Write data row
                writer.writerow(data)
    
    def read_data(self) -> Tuple[List[str], List[List[Any]]]:
        """
        Read all data from the CSV file.
        
        Returns:
            Tuple of (header, data_rows)
        """
        if not os.path.exists(self.filename):
            return [], []
            
        header = []
        data = []
        
        with open(self.filename, 'r', newline='') as f:
            reader = csv.reader(f)
            header = next(reader, [])
            data = [row for row in reader]
            
        return header, data