# --- 1. Environment Setup ---
import pandas as pd
import numpy as np
import time
from urllib.parse import urlsplit
import joblib

# Scikit-learn and Qiskit imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC

# Set a random seed
np.random.seed(42)

# --- 2. Data Loading from Local File ---
print("Loading dataset from local CSV file...")

# Define the path to your local dataset
DATASET_PATH = 'data/phishing-url-dataset.csv'

try:
    # Load the entire dataset using pandas
    df = pd.read_csv(DATASET_PATH)

    # --- FIX: Create the 'HttpsInURL' feature dynamically ---
    # This line was missing. It creates the required column from the URL data.
    df['HttpsInURL'] = df['URL'].apply(lambda x: 1 if str(x).startswith('https') else 0)
    
    # The 'label' column is our target (y), everything else is a feature (X).
    # The 'URL' column is also dropped as it's not a direct feature for the model.
    X_raw = df.drop(['label', 'URL'], axis=1)
    y_raw = df['label']
    
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print(f"Error: Dataset file not found at '{DATASET_PATH}'.")
    print("Please make sure you have downloaded the dataset and placed it in the correct folder.")
    exit()

# --- 3. Feature Selection and Preprocessing ---
# List of 8 numerical features used in the original project for the VQC demo
numerical_features_for_demo = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'HttpsInURL',
    'URLSimilarityIndex', 'CharContinuationRate', 'URLCharProb', 'DomainTitleMatchScore'
]
num_features = len(numerical_features_for_demo)

# Create the final feature set X from the selected columns
# This line will no longer cause an error
X = X_raw[numerical_features_for_demo]

# Ensure y is a flat numpy array for scikit-learn compatibility
y = y_raw.values.ravel()

# Scale features to a suitable range for quantum encoding
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)

# --- 4. Quantum Model (VQC) Implementation ---
# Increase this number for better accuracy, but be aware of longer training times
num_train_samples_q = 100
X_train_q = X_train[:num_train_samples_q]
y_train_q = y_train[:num_train_samples_q]

print(f"\nTraining VQC on {num_train_samples_q} samples... (This may be slow)")

# VQC Architecture
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
optimizer = COBYLA(maxiter=100)
sampler = StatevectorSampler() 

vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer
)

# Train VQC
start_time = time.time()
vqc.fit(X_train_q, y_train_q)
print(f"VQC Training Time: {time.time() - start_time:.2f} seconds")

# --- 5. Save the Model Weights and Scaler ---
print("\nSaving model weights and scaler to disk...")
np.save('vqc_weights.npy', vqc.weights)
joblib.dump(scaler, 'scaler.joblib')
print("Files 'vqc_weights.npy' and 'scaler.joblib' saved successfully!")

print("\n--- Training Script Finished ---")