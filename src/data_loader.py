# src/data_loader.py

import os
import wfdb

def load_ecg_data(person_folder):
    data = {}
    for signal in data.values():
        print(f"Signal shape: {signal.shape}")

    for filename in os.listdir(person_folder):
        if filename.endswith('.dat'):
            record_name = os.path.splitext(filename)[0]
            record_path = os.path.join(person_folder, record_name)
            try:
                # Read the record
                record = wfdb.rdrecord(record_path)
                signal = record.p_signal
                data[record_name] = signal
            except Exception as e:
                print(f"Error reading {record_path}: {e}")
    return data
