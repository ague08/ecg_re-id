# ECG Re-identification Project

This project aims to re-identify users based on ECG data using the **Biometric Human Identification ECG Database (ecgiddb)**.

## Project Structure

- `data/` - Contains the ECG data, split into anonymized and identified datasets.
- `src/` - Contains the source code for data loading, preprocessing, feature extraction, model training, and re-identification.
- `requirements.txt` - Python dependencies.
- `README.md` - Project documentation.

## Getting Started

1. Clone the repository.
2. Install the dependencies: `pip install -r requirements.txt`.
3. Run the code: `python src/main.py`.

## Data Preparation
- The ECG data is divided into two categories:
  - `anonymized/` - 80% of the dataset used for training.
  - `identified/` - 20% of the dataset used for testing and re-identification.
