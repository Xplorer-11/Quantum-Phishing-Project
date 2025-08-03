# --- 1. Imports ---
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from urllib.parse import urlsplit
import io
from PIL import Image
import os
from openai import OpenAI
from typing import List

# Qiskit and Scikit-learn related imports
from qiskit_machine_learning.exceptions import QiskitMachineLearningError
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit.primitives import StatevectorSampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_algorithms.optimizers import COBYLA

# Computer Vision imports
import torch
import torch.nn as nn
from torchvision import models, transforms

# NEW: Import for the database module
import database

# --- 2. API and Model Setup ---
app = FastAPI(
    title="Phish-VQC Prediction API",
    description="API for phishing detection, screenshot analysis, history, and AI support."
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# NEW: Initialize the database on application startup
@app.on_event("startup")
def on_startup():
    database.init_db()

# --- 3. Computer Vision Model Setup ---
class ImageClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(ImageClassifier, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

image_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
vision_model = ImageClassifier()
vision_model.eval()

# --- 4. QML Model Loading and Setup ---
numerical_features_for_demo = [
    'URLLength', 'DomainLength', 'IsDomainIP', 'HttpsInURL',
    'URLSimilarityIndex', 'CharContinuationRate', 'URLCharProb', 'DomainTitleMatchScore'
]
num_features = len(numerical_features_for_demo)

def load_trained_vqc():
    try:
        weights = np.load('vqc_weights.npy')
        feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')
        ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
        sampler = StatevectorSampler()
        optimizer = COBYLA(maxiter=1)
        vqc = VQC(sampler=sampler, feature_map=feature_map, ansatz=ansatz, optimizer=optimizer, initial_point=weights)
        dummy_x = np.zeros((1, num_features))
        dummy_y = np.array([[1, 0]])
        vqc.fit(dummy_x, dummy_y)
        return vqc
    except FileNotFoundError:
        return None

vqc_model = load_trained_vqc()
try:
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    scaler = None

# --- 5. AI Chatbot Setup ---
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key) if api_key else None
if not client:
    print("Warning: OPENAI_API_KEY environment variable not set. Chatbot will not work.")

CHATBOT_SYSTEM_PROMPT = """
You are "Phish-Assist," a friendly and calm cybersecurity assistant. Your purpose is to help users who believe they have interacted with a phishing website. Provide clear, step-by-step, actionable advice. Do not go off-topic. Guide them through: 1. Password Change, 2. Enable 2FA, 3. Financial Monitoring, 4. Malware Scan, 5. Reporting the Site. Always be reassuring and use simple, numbered lists.
"""

# --- 6. Helper Functions & API Endpoints ---
def extract_features_from_url(url: str):
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    features = {}
    try:
        parts = urlsplit(url)
        hostname = parts.hostname or ""
    except Exception:
        return pd.DataFrame([dict.fromkeys(numerical_features_for_demo, 0)])
    features['URLLength'] = len(url)
    features['DomainLength'] = len(parts.netloc)
    is_ip = 1 if hostname and all(c.isdigit() or c == '.' for c in hostname) and hostname.count('.') == 3 else 0
    features['IsDomainIP'] = is_ip
    features['HttpsInURL'] = 1 if parts.scheme == 'https' else 0
    features.update({k: 0.5 for k in ['URLSimilarityIndex', 'CharContinuationRate', 'DomainTitleMatchScore']})
    features['URLCharProb'] = -0.5
    final_features = {key: features.get(key, 0) for key in numerical_features_for_demo}
    return pd.DataFrame([final_features])

class URLItem(BaseModel):
    url: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]

@app.post("/predict")
async def predict_url(item: URLItem):
    if not vqc_model or not scaler:
        return {"error": "Model not loaded."}
    input_url = item.url
    feature_vector = extract_features_from_url(input_url)
    scaled_vector = scaler.transform(feature_vector)
    try:
        prediction = vqc_model.predict(scaled_vector)
        predicted_class = np.argmax(prediction)
        result = 'Legitimate' if predicted_class == 1 else 'Phishing'
        # Log the prediction to the database
        database.log_prediction(input_url, result)
        return {"prediction": result, "url": input_url}
    except Exception as e:
        return {"error": f"An error occurred during prediction: {e}"}

@app.post("/predict-image")
async def predict_image(file: UploadFile = File(...)):
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return {"error": "Invalid image file."}
    image_tensor = image_transforms(image).unsqueeze(0)
    with torch.no_grad():
        outputs = vision_model(image_tensor)
        _, predicted_idx = torch.max(outputs, 1)
    prediction_result = "Legitimate" if predicted_idx.item() == 1 else "Phishing"
    return {"filename": file.filename, "prediction": prediction_result}

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    if not client:
        return {"error": "OpenAI API key is not configured on the server."}
    try:
        messages_with_system_prompt = [{"role": "system", "content": CHATBOT_SYSTEM_PROMPT}] + [msg.dict() for msg in request.messages]
        completion = client.chat.completions.create(model="gpt-3.5-turbo", messages=messages_with_system_prompt)
        return completion.choices[0].message
    except Exception as e:
        return {"error": f"An error occurred with the AI model: {e}"}

# --- NEW: Endpoints for History and Analytics ---
@app.get("/history")
async def get_history():
    """Endpoint to get recent prediction history."""
    return database.get_recent_history()

@app.get("/stats")
async def get_statistics():
    """Endpoint to get phishing statistics."""
    return database.get_stats()

@app.get("/")
def read_root():
    """Root endpoint for health check."""
    return {"status": "Phish-VQC API is running"}