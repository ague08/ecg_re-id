# src/feature_extraction.py
import numpy as np
import neurokit2 as nk
import scipy
from scipy.fft import fft
from scipy.stats import entropy
import pywt

def extract_heartbeat_features(signal, fs=500):
    """
    Extract features from individual heartbeats (R-peak to R-peak).
    
    Args:
        signal (array): The filtered ECG signal.
        fs (int): Sampling frequency.
    
    Returns:
        dict: A dictionary containing the extracted heartbeat features.
    """
    # Detect R-peaks
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    
    # Extract R-R intervals
    rr_intervals = np.diff(rpeaks['ECG_R_Peaks']) / fs  # Convert to seconds
    rr_mean = np.mean(rr_intervals) if len(rr_intervals) > 0 else 0
    rr_std = np.std(rr_intervals) if len(rr_intervals) > 0 else 0

    # Extract HRV features
    hrv_features = nk.hrv_time(rpeaks, sampling_rate=fs)
    
    features = {
        'mean_rr': rr_mean,
        'std_rr': rr_std,
        'hrv_sdnn': hrv_features.get('HRV_SDNN', [0])[0],
        'hrv_rmssd': hrv_features.get('HRV_RMSSD', [0])[0],
    }
    
    return features

def extract_pqrst_fragments(signal, rpeaks, fs=500, fragment_length=250):
    """
    Extract PQRST-fragments synchronized with R-peaks.
    
    Args:
        signal (array): The filtered ECG signal.
        rpeaks (dict): Dictionary containing R-peak indices.
        fs (int): Sampling frequency.
        fragment_length (int): Number of samples per fragment.
    
    Returns:
        list: List of PQRST-fragments.
    """
    fragments = []
    r_indices = rpeaks['ECG_R_Peaks']
    half_fragment = fragment_length // 2

    for i in range(len(r_indices)):
        start = r_indices[i] - half_fragment
        end = r_indices[i] + half_fragment
        if start < 0 or end > len(signal):
            continue  # Skip fragments that exceed signal boundaries
        fragment = signal[start:end]
        # Normalize fragment
        fragment = fragment - np.mean(fragment)
        fragments.append(fragment)
    
    return fragments

def extract_advanced_features_from_ecg(signal, fs=500):
    """
    Extract advanced features from an ECG signal.
    
    Args:
        signal (array): The filtered ECG signal.
        fs (int): Sampling frequency.
    
    Returns:
        dict: A dictionary containing advanced features.
    """
    features = extract_heartbeat_features(signal, fs)
    
    # Fourier Transform - frequency-domain features
    fft_signal = np.abs(fft(signal))[:len(signal) // 2]
    features.update({
        'fft_mean': np.mean(fft_signal),
        'fft_std': np.std(fft_signal),
        'fft_entropy': entropy(fft_signal)
    })
    
    # Wavelet Transform - time-frequency analysis
    coeffs, _ = pywt.cwt(signal, np.arange(1, 31), 'gaus1')
    wavelet_energy = np.sum(np.abs(coeffs)**2, axis=1)
    features.update({
        'wavelet_mean_energy': np.mean(wavelet_energy),
        'wavelet_std_energy': np.std(wavelet_energy)
    })
    
    return features

def extract_pqrst_features(signal, fs=500):
    """
    Extract PQRST-fragment features from the ECG signal.
    
    Args:
        signal (array): The filtered ECG signal.
        fs (int): Sampling frequency.
    
    Returns:
        list: List of feature vectors from PQRST-fragments.
    """
    _, rpeaks = nk.ecg_peaks(signal, sampling_rate=fs)
    fragments = extract_pqrst_fragments(signal, rpeaks, fs)
    fragment_features = []
    
    for fragment in fragments:
        # Feature extraction per fragment can be expanded
        # For simplicity, using raw fragment as feature
        fragment_features.append(fragment)
    
    return fragment_features


# src/feature_extraction.py (continued)

def extract_morphological_features(signal, rpeaks, fs=500):
    """
    Extract morphological features from PQRST-fragments.
    
    Args:
        signal (array): The filtered ECG signal.
        rpeaks (dict): Dictionary containing R-peak indices.
        fs (int): Sampling frequency.
    
    Returns:
        list: List of morphological feature vectors.
    """
    morphological_features = []
    pqrst_fragments = extract_pqrst_fragments(signal, rpeaks, fs)
    
    for fragment in pqrst_fragments:
        # Detect peaks within the fragment
        ppeaks, _ = scipy.signal.find_peaks(fragment, distance=20)
        tpeaks, _ = scipy.signal.find_peaks(-fragment, distance=20)  # Invert signal for T-peaks
        
        # Calculate wave amplitudes
        p_amplitude = np.max(fragment[ppeaks]) if ppeaks.size > 0 else 0
        q_amplitude = np.min(fragment[:np.argmax(fragment)]) if np.argmax(fragment) > 0 else 0
        r_amplitude = np.max(fragment)
        s_amplitude = np.min(fragment[np.argmax(fragment):]) if np.argmax(fragment) < len(fragment)-1 else 0
        t_amplitude = np.max(fragment[tpeaks]) if tpeaks.size > 0 else 0
        
        # Calculate wave durations (simplified as time between peaks)
        pr_interval = (ppeaks[0] - rpeaks['ECG_R_Peaks'][0])/fs if ppeaks.size > 0 else 0
        qt_interval = (tpeaks[-1] - rpeaks['ECG_R_Peaks'][-1])/fs if tpeaks.size > 0 and len(rpeaks['ECG_R_Peaks']) > 0 else 0

        
        features = [
            p_amplitude,
            q_amplitude,
            r_amplitude,
            s_amplitude,
            t_amplitude,
            pr_interval,
            qt_interval
        ]
        
        morphological_features.append(features)
    
    return morphological_features


# src/feature_extraction.py (continued)

def extract_non_linear_features(signal):
    """
    Extract nonlinear features from the ECG signal.
    
    Args:
        signal (array): The filtered ECG signal.
    
    Returns:
        dict: Dictionary of nonlinear features.
    """
    # Approximate Entropy
    approx_entropy = nk.entropy_approximate(signal)
    
    # Sample Entropy
    sample_entropy = nk.entropy_sample(signal)
    
    return {
        'approx_entropy': approx_entropy,
        'sample_entropy': sample_entropy
    }

def extract_frequency_features(signal, fs=500):
    """
    Extract frequency-domain features from the ECG signal.
    
    Args:
        signal (array): The filtered ECG signal.
        fs (int): Sampling frequency.
    
    Returns:
        dict: Dictionary of frequency-domain features.
    """
    freqs, psd = signal_welch(signal, fs)
    dominant_freq = freqs[np.argmax(psd)]
    total_power = np.sum(psd)
    
    return {
        'dominant_freq': dominant_freq,
        'total_power': total_power
    }

def signal_welch(signal, fs=500):
    """
    Compute the Power Spectral Density using Welch's method.
    
    Args:
        signal (array): The input signal.
        fs (int): Sampling frequency.
    
    Returns:
        tuple: Frequencies and power spectral density.
    """
    from scipy.signal import welch
    freqs, psd = welch(signal, fs)
    return freqs, psd
