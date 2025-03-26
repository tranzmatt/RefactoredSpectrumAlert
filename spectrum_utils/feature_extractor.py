#!/usr/bin/env python3
"""
Feature extraction utilities for SpectrumAlert.
"""

from typing import List, Optional

import numpy as np


class FeatureExtractor:
    """Extracts features from IQ data samples."""
    
    @staticmethod
    def extract_features(iq_data: np.ndarray, target_num_features: Optional[int] = None) -> List[float]:
        """
        Extract features from IQ data for model prediction.
        
        Args:
            iq_data: Complex IQ data samples
            target_num_features: Optional target number of features to return
            
        Returns:
            List of extracted features
        """
        try:
            # Separate real and imaginary parts
            I = np.real(iq_data)
            Q = np.imag(iq_data)
            
            # Amplitude and phase
            amplitude = np.sqrt(I ** 2 + Q ** 2)
            phase = np.unwrap(np.angle(iq_data))
            
            # Basic statistics
            mean_amplitude = np.mean(amplitude)
            std_amplitude = np.std(amplitude)
            
            # Calculate higher-order statistics safely
            features = FeatureExtractor._calculate_higher_order_stats(
                amplitude, phase, mean_amplitude, std_amplitude
            )
            
            # Adjust feature count if required
            if target_num_features is not None:
                if len(features) < target_num_features:
                    # Pad with zeros if we have fewer features than expected
                    features.extend([0.0] * (target_num_features - len(features)))
                elif len(features) > target_num_features:
                    # Truncate if we have more features than expected
                    features = features[:target_num_features]
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            # Return zeros if feature extraction fails
            return [0.0] * (target_num_features if target_num_features is not None else 9)
    
    @staticmethod
    def _calculate_higher_order_stats(
        amplitude: np.ndarray, 
        phase: np.ndarray, 
        mean_amplitude: float, 
        std_amplitude: float
    ) -> List[float]:
        """
        Calculate higher-order statistics from signal components.
        
        Args:
            amplitude: Signal amplitude
            phase: Signal phase
            mean_amplitude: Mean amplitude
            std_amplitude: Standard deviation of amplitude
            
        Returns:
            List of calculated features
        """
        features = []
        
        # Add basic features
        features.extend([mean_amplitude, std_amplitude])
        
        # FFT of the signal
        try:
            fft_values = np.fft.fft(amplitude)
            fft_magnitude = np.abs(fft_values)
            
            # FFT statistics
            mean_fft = np.mean(fft_magnitude)
            std_fft = np.std(fft_magnitude)
            features.extend([mean_fft, std_fft])
        except Exception:
            features.extend([0.0, 0.0])  # Default values on error
        
        # Skewness and kurtosis of amplitude (safely)
        try:
            if std_amplitude > 1e-10:
                skew_amplitude = np.mean((amplitude - mean_amplitude) ** 3) / (std_amplitude ** 3)
                kurt_amplitude = np.mean((amplitude - mean_amplitude) ** 4) / (std_amplitude ** 4)
            else:
                skew_amplitude = 0.0
                kurt_amplitude = 0.0
            features.extend([skew_amplitude, kurt_amplitude])
        except Exception:
            features.extend([0.0, 0.0])  # Default values on error
            
        # Phase statistics
        try:
            mean_phase = np.mean(phase)
            std_phase = np.std(phase)
            
            if std_phase > 1e-10:
                skew_phase = np.mean((phase - mean_phase) ** 3) / (std_phase ** 3)
                kurt_phase = np.mean((phase - mean_phase) ** 4) / (std_phase ** 4)
            else:
                skew_phase = 0.0
                kurt_phase = 0.0
            features.extend([skew_phase, kurt_phase])
        except Exception:
            features.extend([0.0, 0.0])  # Default values on error
            
        # Cyclostationary features (simplified)
        try:
            if len(amplitude) > 1:
                autocorr = np.correlate(amplitude, amplitude, mode='full')
                mid_point = len(autocorr) // 2
                cyclo_autocorr = np.abs(autocorr[mid_point:]).mean()
            else:
                cyclo_autocorr = 0.0
            features.append(cyclo_autocorr)
        except Exception:
            features.append(0.0)  # Default value on error
            
        return features
    
    @staticmethod
    def calculate_signal_strength(iq_data: np.ndarray) -> float:
        """
        Calculate signal strength in dB from IQ samples.
        
        Args:
            iq_data: Complex IQ data samples
            
        Returns:
            Signal strength in dB
        """
        try:
            amplitude = np.abs(iq_data)
            # Use mean squared amplitude for power calculation
            signal_power = np.mean(amplitude ** 2)
            
            # Avoid log of zero
            if signal_power > 1e-10:
                signal_strength_db = 10 * np.log10(signal_power)
            else:
                signal_strength_db = -100.0  # Default low value
                
            return signal_strength_db
            
        except Exception as e:
            print(f"Error calculating signal strength: {e}")
            return -100.0  # Default low value on error