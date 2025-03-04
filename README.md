
# ECG De-anonymization Attacks Project 

This project aims to re-identify users based on ECG data using the **Biometric Human Identification ECG Databases (ECG-IDB+MIT-BIH)**.

## Project Structure

- `data/` - Contains the ECG data, split into anonymized and identified datasets.
- `src/` - Contains the source code for data loading, preprocessing, feature extraction, model training, and re-identification.
- `requirements.txt` - Python dependencies.
- `README.md` - Project documentation.

## Getting Started

1. Install the dependencies: `pip install -r requirements.txt`.
2. Prepare datasets and paths.
3. Run the code: `python src/main.py`.

## Data Preparation
- The ECG data is divided into two categories:
  - `anonymized/` - 75% of the dataset's records were used as de-identified.
  - `identified/` - 25% of the dataset's records were used for re-identification.
- 3 different splits were done for cross-validation.
=======
# ecg_re-id

