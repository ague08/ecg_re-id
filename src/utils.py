# src/utils.py
import numpy as np
import json
import os

def normalize_vector(vector):
    """
    Normalize a feature vector to have zero mean and unit variance.
    
    Args:
        vector (array): Input feature vector.
    
    Returns:
        array: Normalized feature vector.
    """
    mean = np.mean(vector)
    std = np.std(vector) if np.std(vector) != 0 else 1
    return (vector - mean) / std

def save_fingerprints(fingerprints, filepath):
    """
    Save fingerprints to a JSON file.
    
    Args:
        fingerprints (dict): Dictionary of fingerprints.
        filepath (str): Path to the JSON file.
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump({k: v.tolist() for k, v in fingerprints.items()}, f)

def load_fingerprints(filepath):
    """
    Load fingerprints from a JSON file.
    
    Args:
        filepath (str): Path to the JSON file.
    
    Returns:
        dict: Dictionary of fingerprints.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return {k: np.array(v) for k, v in data.items()}
