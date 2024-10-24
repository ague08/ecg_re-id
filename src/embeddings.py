# src/embeddings.py
import numpy as np
from data_loader import load_ecg_data
from preprocessing import preprocess_signal
from feature_extraction import extract_pqrst_features
import neurokit2 as nk

def create_heartbeat_embeddings(person_folder, cnn_model, fs=500, fragment_length=250):
    """
    Create CNN embeddings for each heartbeat in the ECG recordings of a person.
    
    Args:
        person_folder (str): Path to the person's folder.
        cnn_model (tf.keras.Model): Trained CNN model.
        fs (int): Sampling frequency.
        fragment_length (int): Number of samples per fragment.
    
    Returns:
        np.array: Combined embeddings from all heartbeats.
    """
    embeddings = []
    data = load_ecg_data(person_folder)
    
    for signal in data.values():
        # Preprocess the signal
        filtered_signal = preprocess_signal(signal[:, 1], fs)
        
        # Extract PQRST-fragments
        pqrst_fragments = extract_pqrst_features(filtered_signal, fs)
        
        for fragment in pqrst_fragments:
            if len(fragment) == fragment_length:
                # Reshape for CNN input: (1, 250, 1)
                fragment_input = np.expand_dims(np.expand_dims(fragment, axis=0), axis=-1)
                embedding = cnn_model.predict(fragment_input, verbose=0)
                embeddings.append(embedding[0])
    
    if embeddings:
        # Average embeddings to create a single fingerprint
        final_embedding = np.mean(embeddings, axis=0)
    else:
        final_embedding = np.zeros(cnn_model.output_shape[1:])
    
    return final_embedding
