# train_and_save_model.py

# --- 1. Environment Setup ---
import pandas as pd
import numpy as np
import time
from urllib.parse import urlsplit
import joblib # Still needed for the scaler

# Scikit-learn and Qiskit imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
# CORRECTED: Use the recommended V2 primitive
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from ucimlrepo import fetch_ucirepo

# Set a random seed
np.random.seed(42)

# --- 2. Data Loading and Feature Selection ---
print("Fetching dataset...")
phiusiil_phishing_url = fetch_ucirepo(id=967)
X_raw = phiusiil_phishing_url.data.features
y_raw = phiusiil_phishing_url.data.targets
print("Dataset fetched.")

numerical_features_for_demo = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'HttpsInURL',
    'URLSimilarityIndex', 'CharContinuationRate', 'URLCharProb', 'DomainTitleMatchScore'
]
num_features = len(numerical_features_for_demo)

# CORRECTED: Use .loc to avoid the SettingWithCopyWarning
X_raw.loc[:, 'HttpsInURL'] = X_raw['URL'].apply(lambda x: 1 if urlsplit(x).scheme == 'https' else 0)
X = X_raw[numerical_features_for_demo]
y = y_raw.values.ravel()

# --- 3. Data Preprocessing ---
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_train, _, y_train, _ = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)

# --- 4. Quantum Model (VQC) Implementation ---
num_train_samples_q = 100
X_train_q = X_train[:num_train_samples_q]
y_train_q = y_train[:num_train_samples_q]

print(f"\nTraining VQC on {num_train_samples_q} samples... (This will be slow)")

# VQC Architecture
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
optimizer = COBYLA(maxiter=100)
# CORRECTED: Use the non-deprecated sampler
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

# --- 5. Save the Model WEIGHTS and Scaler ---
print("\nSaving model weights and scaler to disk...")
# CORRECTED: Save only the weights, not the whole object, using NumPy
np.save('vqc_weights.npy', vqc.weights)
joblib.dump(scaler, 'scaler.joblib')
print("Files 'vqc_weights.npy' and 'scaler.joblib' saved successfully!")

print("\n--- Training Script Finished ---")