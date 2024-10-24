# src/main.py
import os
import numpy as np
from data_loader import load_ecg_data
from preprocessing import augment_signal, preprocess_signal
from feature_extraction import extract_advanced_features_from_ecg, extract_pqrst_features, extract_pqrst_fragments
from embeddings import create_heartbeat_embeddings
from model import create_improved_cnn_model
from evaluation import de_anonymize, calculate_accuracy
from utils import normalize_vector

from keras.utils import to_categorical
from feature_extraction import extract_morphological_features
import neurokit2 as nk



# src/main.py (continued)
from feature_extraction import extract_non_linear_features, extract_frequency_features

# def create_combined_fingerprint(person_folder, cnn_model, fs=500):
#     """
#     Create a combined fingerprint by integrating CNN embeddings with advanced, morphological, nonlinear, and frequency features.
    
#     Args:
#         person_folder (str): Path to the person's folder.
#         cnn_model (tf.keras.Model): Trained CNN model.
#         fs (int): Sampling frequency.
    
#     Returns:
#         np.array: Combined fingerprint vector.
#     """
#     # Generate CNN embeddings
#     cnn_embedding = create_heartbeat_embeddings(person_folder, cnn_model, fs)
    
#     # Load and preprocess data
#     data = load_ecg_data(person_folder)
    
#     all_advanced_features = []
#     all_morphological_features = []
#     all_non_linear_features = []
#     all_frequency_features = []
    
#     for signal in data.values():
#         # Preprocess the signal
#         filtered_signal = preprocess_signal(signal[:, 1], fs)
        
#         # Extract advanced features
#         advanced_features = extract_advanced_features_from_ecg(filtered_signal, fs)
#         advanced_features_vector = np.array(list(advanced_features.values()))
#         advanced_features_vector = normalize_vector(advanced_features_vector)
#         all_advanced_features.append(advanced_features_vector)
        
#         # Extract morphological features
#         _, rpeaks = nk.ecg_peaks(filtered_signal, sampling_rate=fs)
#         morphological_features = extract_morphological_features(filtered_signal, rpeaks, fs)
#         if morphological_features:
#             mean_morphological = np.mean(morphological_features, axis=0)
#         else:
#             mean_morphological_features = np.zeros(7)  # Assuming 7 morphological features
#         all_morphological_features.append(mean_morphological_features)
        
#         # Extract nonlinear features
#         nonlinear_features = extract_non_linear_features(filtered_signal)
#         nonlinear_features_vector = np.array(list(nonlinear_features.values()))
#         nonlinear_features_vector = normalize_vector(nonlinear_features_vector)
#         all_non_linear_features.append(nonlinear_features_vector)
        
#         # Extract frequency features
#         frequency_features = extract_frequency_features(filtered_signal, fs)
#         frequency_features_vector = np.array(list(frequency_features.values()))
#         frequency_features_vector = normalize_vector(frequency_features_vector)
#         all_frequency_features.append(frequency_features_vector)
    
#     if all_advanced_features:
#         # Average advanced features across all recordings
#         mean_advanced_features = np.mean(all_advanced_features, axis=0)
#     else:
#         mean_advanced_features = np.zeros(advanced_features_vector.shape)
    
#     if all_morphological_features:
#         # Average morphological features across all recordings
#         mean_morphological_features = np.mean(all_morphological_features, axis=0)
#     else:
#         mean_morphological_features = np.zeros(7)
    
#     if all_non_linear_features:
#         # Average nonlinear features across all recordings
#         mean_nonlinear_features = np.mean(all_non_linear_features, axis=0)
#     else:
#         mean_nonlinear_features = np.zeros(2)  # Assuming 2 nonlinear features
     
#     if all_frequency_features:
#         # Average frequency features across all recordings
#         mean_frequency_features = np.mean(all_frequency_features, axis=0)
#     else:
#         mean_frequency_features = np.zeros(2)  # Assuming 2 frequency features
    
#     # Combine all features: CNN embeddings + advanced features + morphological features + nonlinear features + frequency features
#     combined_fingerprint = np.concatenate([
#         cnn_embedding,
#         mean_advanced_features,
#         mean_morphological_features,
#         mean_nonlinear_features,
#         mean_frequency_features
#     ])
    
#     return combined_fingerprint

def create_combined_fingerprint(person_folder, cnn_model, fs=500):
    cnn_embedding = create_heartbeat_embeddings(person_folder, cnn_model, fs)
    return cnn_embedding


# src/main.py (continued)

def prepare_training_data(train_dir, cnn_model, fs=500, fragment_length=250, augment=False):
    """
    Prepare training data by extracting PQRST-fragments and labels, with optional augmentation.
    
    Args:
        train_dir (str): Path to the training data directory.
        cnn_model (tf.keras.Model): CNN model.
        fs (int): Sampling frequency.
        fragment_length (int): Number of samples per fragment.
        augment (bool): Whether to apply data augmentation.
    
    Returns:
        X_train (array): Training data.
        y_train (array): Training labels.
        unique_labels (list): List of unique labels.
    """
    X_train = []
    y_train = []
    
    for person in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person)
        if os.path.isdir(person_folder):
            data = load_ecg_data(person_folder)
            for signal in data.values():
                filtered_signal = preprocess_signal(signal[:, 1], fs)
                pqrst_fragments = extract_pqrst_features(filtered_signal, fs)
                
                for fragment in pqrst_fragments:
                    if len(fragment) == fragment_length:
                        if augment:
                            fragment = augment_signal(fragment, fs)
                        fragment_input = np.expand_dims(np.expand_dims(fragment, axis=0), axis=-1)
                        X_train.append(fragment_input)
                        y_train.append(person)  # Assuming person IDs are labels
    
    # Convert to numpy arrays
    X_train = np.vstack(X_train)
    # Encode labels
    unique_labels = sorted(list(set(y_train)))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    y_train_encoded = np.array([label_to_index[label] for label in y_train])
    y_train_categorical = to_categorical(y_train_encoded, num_classes=len(unique_labels))
    
    return X_train, y_train_categorical, unique_labels


# def main(train_dir, test_dir, model_type='cnn', epochs=50, batch_size=32):
#     """
#     Main function to perform ECG de-anonymization.
    
#     Args:
#         train_dir (str): Path to the training data directory.
#         test_dir (str): Path to the test data directory.
#         model_type (str): Type of model to use ('cnn' or 'cnn_lstm').
#         epochs (int): Number of training epochs.
#         batch_size (int): Training batch size.
#     """
#     # Create the model
#     if model_type == 'cnn':
#         input_shape = (250, 1)  # PQRST-fragment length and channels
#         cnn_model = create_improved_cnn_model(input_shape)
#     elif model_type == 'cnn_lstm':
#         input_shape = (5, 250, 1)  # Example: sequences of 5 PQRST-fragments
#         cnn_model = create_cnn_lstm_model(input_shape)
#     else:
#         raise ValueError("Unsupported model type. Choose 'cnn' or 'cnn_lstm'.")
    
#     # Prepare training data
#     print("Preparing training data...")
#     X_train, y_train, unique_labels = prepare_training_data(train_dir, cnn_model)
    
#     # Train the model
#     print("Training the model...")
#     cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
#     # Save the trained model
#     model_path = os.path.join('..', 'models', f'{model_type}_model.h5')
#     cnn_model.save(model_path)
#     print(f"Model saved to {model_path}")
    
#     # Create and save training fingerprints
#     train_fingerprints = {}
#     for person in os.listdir(train_dir):
#         person_folder = os.path.join(train_dir, person)
#         if os.path.isdir(person_folder):
#             print(f"Creating fingerprint for {person}...")
#             fingerprint = create_combined_fingerprint(person_folder, cnn_model)
#             train_fingerprints[person] = fingerprint
    
#     # Optionally, save fingerprints to disk
#     from utils import save_fingerprints
#     save_fingerprints(train_fingerprints, '/Users/hamza/Desktop/ecg-ide-db-1.0.0 - de-anonymization folder/code-models/model-2-struct/models/train_fingerprints.json')
    
#     # Create delta primes for test data
#     test_delta_primes = {}
#     for person in os.listdir(test_dir):
#         person_folder = os.path.join(test_dir, person)
#         if os.path.isdir(person_folder):
#             print(f"Creating delta prime for {person}...")
#             delta_prime = create_combined_fingerprint(person_folder, cnn_model)
#             test_delta_primes[person] = delta_prime
    
#     # Perform de-anonymization
#     print("\nDe-anonymization Results:")
#     match_results = de_anonymize(test_delta_primes, train_fingerprints, metric='cosine')  # Using cosine similarity
#     for test_user, predicted_user in match_results.items():
#         print(f"Test User {test_user} matches with Training User {predicted_user}")
    
#     # Calculate accuracy
#     accuracy = calculate_accuracy(match_results)
#     print(f"\nDe-anonymization Accuracy: {accuracy:.2f}%")

# src/main.py (continued)
def main(train_dir, test_dir, model_type='cnn', epochs=50, batch_size=32, augment=False):
    """
    Main function to perform ECG de-anonymization.
    
    Args:
        train_dir (str): Path to the training data directory.
        test_dir (str): Path to the test data directory.
        model_type (str): Type of model to use ('cnn' or 'cnn_lstm').
        epochs (int): Number of training epochs.
        batch_size (int): Training batch size.
        augment (bool): Whether to apply data augmentation.
    """
    # Create the model
    
    input_shape = (250, 1)  # PQRST-fragment length and channels
    cnn_model = create_improved_cnn_model(input_shape)
    
    
    # Prepare training data
    print("Preparing training data...")
    X_train, y_train, unique_labels = prepare_training_data(train_dir, cnn_model, augment=augment)
    
    # Train the model
    print("Training the model...")
    cnn_model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
    
    # Save the trained model
    model_path = os.path.join('..', 'models', f'{model_type}_model.h5')
    cnn_model.save(model_path)
    print(f"Model saved to {model_path}")
    
    # Create and save training fingerprints
    train_fingerprints = {}
    for person in os.listdir(train_dir):
        person_folder = os.path.join(train_dir, person)
        if os.path.isdir(person_folder):
            print(f"Creating fingerprint for {person}...")
            fingerprint = create_combined_fingerprint(person_folder, cnn_model)
            train_fingerprints[person] = fingerprint
    
    # Optionally, save fingerprints to disk
    # Optionally, save fingerprints to disk
    from utils import save_fingerprints
    save_fingerprints(train_fingerprints, '/Users/hamza/Desktop/ecg-ide-db-1.0.0 - de-anonymization folder/code-models/code 2 (88acc)- model-2-structured/models/train_fingerprints.json')

    
    # Create delta primes for test data
    test_delta_primes = {}
    for person in os.listdir(test_dir):
        person_folder = os.path.join(test_dir, person)
        if os.path.isdir(person_folder):
            print(f"Creating delta prime for {person}...")
            delta_prime = create_combined_fingerprint(person_folder, cnn_model)
            test_delta_primes[person] = delta_prime
    
    # Perform de-anonymization
    print("\nDe-anonymization Results:")
    match_results = de_anonymize(test_delta_primes, train_fingerprints, metric='cosine')  # Using cosine similarity
    for test_user, predicted_user in match_results.items():
        print(f"Test User {test_user} matches with Training User {predicted_user}")
    
    # Calculate accuracy
    accuracy = calculate_accuracy(match_results)
    print(f"\nDe-anonymization Accuracy: {accuracy:.2f}%")


if __name__ == '__main__':
    train_dir = '/Users/hamza/Desktop/ecg-ide-db-1.0.0 - de-anonymization folder/code-models/code 2 (88acc)- model-2-structured/data/train/'
    test_dir = '/Users/hamza/Desktop/ecg-ide-db-1.0.0 - de-anonymization folder/code-models/code 2 (88acc)- model-2-structured/data/test/'
    main(train_dir, test_dir, model_type='cnn')  # Choose 'cnn_lstm' if using CNN-LSTM model
