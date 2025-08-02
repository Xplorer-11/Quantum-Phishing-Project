# --- 1. Imports ---
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlsplit
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

# Imports needed to rebuild the VQC model
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA

# --- 2. API and Model Setup ---
app = FastAPI(
    title="Phish-VQC Prediction API",
    description="API for predicting phishing URLs using a VQC model."
)

# Configure CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Define the feature list that the model expects
numerical_features_for_demo = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'HttpsInURL',
    'URLSimilarityIndex', 'CharContinuationRate', 'URLCharProb', 'DomainTitleMatchScore'
]
num_features = len(numerical_features_for_demo)

# --- 3. Function to Load Model and Weights ---
def load_trained_vqc():
    """
    Reconstructs the VQC model, loads weights, and performs a "dummy fit"
    to prepare the model for prediction.
    """
    try:
        weights = np.load('vqc_weights.npy')
        
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')
        ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
        sampler = StatevectorSampler()
        optimizer = COBYLA(maxiter=0) 
        
        vqc = VQC(
            sampler=sampler,
            feature_map=feature_map,
            ansatz=ansatz,
            optimizer=optimizer,
            initial_point=weights
        )
        
        print("Performing dummy fit to prepare model for prediction...")
        dummy_x = np.zeros((1, num_features))
        dummy_y = np.array([[1, 0]])
        
        vqc.fit(dummy_x, dummy_y) 
        
        print("VQC model reconstructed and ready for predictions.")
        return vqc
    except FileNotFoundError:
        print("Error: 'vqc_weights.npy' not found. Please run 'train_and_save_model.py' first.")
        return None

# Load the model and scaler on server startup
vqc_model = load_trained_vqc()
try:
    scaler = joblib.load('scaler.joblib')
    print("Scaler loaded successfully.")
except FileNotFoundError:
    scaler = None
    print("Error: 'scaler.joblib' not found.")

# --- 4. Feature Extraction ---
def extract_features_from_url(url: str):
    features = {}
    try:
        parts = urlsplit(url)
        hostname = parts.hostname if parts.hostname else ""
    except Exception:
        return pd.DataFrame([dict.fromkeys(numerical_features_for_demo, 0)])

    features['URLLength'] = len(url)
    features['DomainLength'] = len(parts.netloc)
    is_ip = 1 if hostname and all(block.isdigit() for block in hostname.split('.') if block) and len(hostname.split('.')) == 4 else 0
    features['IsDomainIP'] = is_ip
    features['HttpsInURL'] = 1 if parts.scheme == 'https' else 0
    
    features['URLSimilarityIndex'] = 0.5
    features['CharContinuationRate'] = 0.5
    features['URLCharProb'] = -0.5
    features['DomainTitleMatchScore'] = 0.5

    final_features = {key: features.get(key, 0) for key in numerical_features_for_demo}
    return pd.DataFrame([final_features])

# --- 5. API Endpoints ---
class URLItem(BaseModel):
    url: str

@app.post("/predict")
async def predict_url(item: URLItem):
    if not vqc_model or not scaler:
        return {"error": "Model or scaler not loaded. Server is not ready."}

    input_url = item.url
    
    feature_vector = extract_features_from_url(input_url)
    scaled_vector = scaler.transform(feature_vector)
    
    try:
        prediction = vqc_model.predict(scaled_vector)
        
        # CORRECTED: The model output is now an array of scores, e.g., [[0.1, 0.9]].
        # We use np.argmax() to find the index of the highest score, which gives us the predicted class (0 or 1).
        predicted_class = np.argmax(prediction)
        
        result = 'Legitimate' if predicted_class == 1 else 'Phishing'
        
        return {"prediction": result, "url": input_url}
    except (QiskitMachineLearningError, Exception) as e:
        return {"error": f"An error occurred during prediction: {e}"}

@app.get("/")
def read_root():
    return {"status": "Phish-VQC API is running"}