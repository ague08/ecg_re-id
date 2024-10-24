# src/preprocessing.py
import pywt
import numpy as np
from scipy.signal import butter, filtfilt
import neurokit2 as nk
import os

def preprocess_signal(signal, fs):
    # Remove baseline wandering
    baseline = nk.ecg_clean(signal, sampling_rate=fs)
    
    # Interpolate baseline to match the signal length if necessary
    if len(baseline) != len(signal):
        baseline = np.interp(np.arange(len(signal)), np.linspace(0, len(signal)-1, len(baseline)), baseline)
    
    corrected_signal = signal - baseline
    return corrected_signal

def apply_bandstop_filter(signal, low, high, fs, order=5):
    """
    Apply a Butterworth bandstop filter.
    
    Args:
        signal (array): Input signal.
        low (float): Low cutoff frequency.
        high (float): High cutoff frequency.
        fs (int): Sampling frequency.
        order (int): Order of the filter.
    
    Returns:
        array: Filtered signal.
    """
    nyq = 0.5 * fs
    low /= nyq
    high /= nyq
    b, a = butter(order, [low, high], btype='bandstop')
    y = filtfilt(b, a, signal)
    return y

def apply_lowpass_filter(signal, cutoff, fs, order=5):
    """
    Apply a Butterworth lowpass filter.
    
    Args:
        signal (array): Input signal.
        cutoff (float): Cutoff frequency.
        fs (int): Sampling frequency.
        order (int): Order of the filter.
    
    Returns:
        array: Filtered signal.
    """
    nyq = 0.5 * fs
    cutoff /= nyq
    b, a = butter(order, cutoff, btype='low')
    y = filtfilt(b, a, signal)
    return y

# src/preprocessing.py (continued)
import random

def augment_signal(signal, fs=500):
    """
    Apply random augmentations to the ECG signal.
    
    Args:
        signal (array): Input ECG signal.
        fs (int): Sampling frequency.
    
    Returns:
        array: Augmented ECG signal.
    """
    # Add Gaussian noise
    noise = np.random.normal(0, 0.01, len(signal))
    augmented_signal = signal + noise
    
    # Randomly shift the signal
    shift = random.randint(-10, 10)
    augmented_signal = np.roll(augmented_signal, shift)
    
    # Randomly scale the signal
    scale = random.uniform(0.9, 1.1)
    augmented_signal = augmented_signal * scale
    
    return augmented_signal
