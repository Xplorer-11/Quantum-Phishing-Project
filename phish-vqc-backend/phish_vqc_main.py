# ==============================================================================
# Appendix A: Corrected Project Codebase for Phish-VQC
# ==============================================================================

# --- 1. Environment Setup ---
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
from urllib.parse import urlsplit

# Scikit-learn for classical ML and preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Qiskit for Quantum ML
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_algorithms.optimizers import COBYLA
from qiskit.primitives import Sampler
from qiskit_machine_learning.algorithms.classifiers import VQC
from qiskit_machine_learning.exceptions import QiskitMachineLearningError

# UCI ML Repo for data fetching
from ucimlrepo import fetch_ucirepo

# Set a random seed for reproducibility
np.random.seed(42)

# --- 2. Data Loading and Feature Selection ---
print("Fetching dataset from UCI Repository...")
try:
    phiusiil_phishing_url = fetch_ucirepo(id=967)
    X_raw = phiusiil_phishing_url.data.features
    y_raw = phiusiil_phishing_url.data.targets
    # The dataset uses 1 for legitimate, 0 for phishing.
    print("Dataset fetched successfully.")
except Exception as e:
    print(f"Failed to fetch dataset. Please check your internet connection or the UCI repository status. Error: {e}")
    exit()

# CORRECTED: Defined the list of numerical features to be used.
# The original code left this list empty. We select 8 features for this demo.
numerical_features_for_demo = [
    'URLLength',
    'DomainLength',
    'IsDomainIP',
    'HttpsInURL',
    'URLSimilarityIndex',
    'CharContinuationRate',
    'URLCharProb',
    'DomainTitleMatchScore'
]
num_features = len(numerical_features_for_demo)
print(f"\nUsing a subset of {num_features} numerical features for this demonstration.")

# Create the HttpsInURL feature which is not in the original dataset but implied by the extractor
X_raw['HttpsInURL'] = X_raw['URL'].apply(lambda x: 1 if urlsplit(x).scheme == 'https' else 0)

X = X_raw[numerical_features_for_demo]
y = y_raw.values.ravel()

# --- 3. Data Preprocessing ---
# Scale features to a suitable range for quantum encoding
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets (80/20 split)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.20, random_state=42, stratify=y)

print(f"\nData preprocessed and split:")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

# --- 4. Classical Model Benchmarking ---
print("\n--- Benchmarking Classical Models ---")

# a) Random Forest
print("\nTraining Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
start_time = time.time()
rf_model.fit(X_train, y_train)
rf_training_time = time.time() - start_time
y_pred_rf = rf_model.predict(X_test)

print(f"RF Training Time: {rf_training_time:.2f} seconds")
print("RF Performance:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_rf):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_rf):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_rf):.4f}")

# b) Support Vector Machine
print("\nTraining Support Vector Machine Classifier...")
svm_model = SVC(kernel='rbf', C=1.0, random_state=42)
start_time = time.time()
svm_model.fit(X_train, y_train)
svm_training_time = time.time() - start_time
y_pred_svm = svm_model.predict(X_test)

print(f"SVM Training Time: {svm_training_time:.2f} seconds")
print("SVM Performance:")
print(f"  Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_svm):.4f}")
print(f"  Recall: {recall_score(y_test, y_pred_svm):.4f}")
print(f"  F1-Score: {f1_score(y_test, y_pred_svm):.4f}")

# --- 5. Quantum Model (VQC) Implementation ---
print("\n--- Implementing Variational Quantum Classifier ---")
# For demonstration purposes, we will use a smaller subset of the data
# as simulating a large dataset is computationally intensive.
num_train_samples_q = 100
num_test_samples_q = 25

X_train_q = X_train[:num_train_samples_q]
y_train_q = y_train[:num_train_samples_q]
X_test_q = X_test[:num_test_samples_q]
y_test_q = y_test[:num_test_samples_q]

print(f"Using a smaller dataset for VQC demonstration:")
print(f"Quantum training set size: {len(X_train_q)}")
print(f"Quantum testing set size: {len(X_test_q)}")

# a) Define VQC Architecture
feature_map = ZZFeatureMap(feature_dimension=num_features, reps=2, entanglement='linear')
ansatz = RealAmplitudes(num_qubits=num_features, reps=3)
optimizer = COBYLA(maxiter=100)
sampler = Sampler()

# Callback function to observe training progress
objective_func_vals = []
def callback_graph(weights, obj_func_eval):
    objective_func_vals.append(obj_func_eval)
    print(f"Iteration {len(objective_func_vals)}: Cost = {obj_func_eval:.4f}", end='\r')

# b) Instantiate and Train VQC
vqc = VQC(
    sampler=sampler,
    feature_map=feature_map,
    ansatz=ansatz,
    optimizer=optimizer,
    callback=callback_graph
)

print("\nTraining VQC... (This will take a significant amount of time)")
start_time = time.time()
vqc.fit(X_train_q, y_train_q)
vqc_training_time = time.time() - start_time
print() # CORRECTED: Add a newline to prevent overwriting the final callback message.

print(f"\nVQC Training Time: {vqc_training_time:.2f} seconds")

# Plot the training progress
plt.figure(figsize=(10, 6))
plt.title("VQC Training Progress")
plt.plot(range(len(objective_func_vals)), objective_func_vals, label="Cost Function")
plt.xlabel("Optimization Iteration")
plt.ylabel("Cost")
plt.legend()
plt.grid(True)
plt.show()

# c) Evaluate VQC
print("\nEvaluating VQC Performance...")
try:
    y_pred_vqc = vqc.predict(X_test_q)
    print("VQC Performance:")
    print(f"  Accuracy: {accuracy_score(y_test_q, y_pred_vqc):.4f}")
    print(f"  Precision: {precision_score(y_test_q, y_pred_vqc, zero_division=0):.4f}")
    print(f"  Recall: {recall_score(y_test_q, y_pred_vqc, zero_division=0):.4f}")
    print(f"  F1-Score: {f1_score(y_test_q, y_pred_vqc, zero_division=0):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test_q, y_pred_vqc)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Phishing', 'Legitimate'], yticklabels=['Phishing', 'Legitimate'])
    plt.title('VQC Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

except QiskitMachineLearningError as e:
    print(f"Could not evaluate VQC. This might happen if the model did not converge well. Error: {e}")


# --- 6. Real-Time Prediction Function ---
print("\n--- Real-Time URL Prediction Function ---")

# CORRECTED: The entire feature extraction function was flawed.
# The original code repeatedly overwrote the 'features' variable instead of assigning key-value pairs to a dictionary.
# This caused a fatal TypeError in the last line of the function.
# The function is now corrected to build a dictionary of features properly.
def extract_features_from_url(url):
    """
    A corrected feature extractor for demonstration.
    NOTE: This function is designed to produce a feature vector compatible with the demo model.
    It extracts lexical features directly from the URL string.
    Features requiring webpage content (e.g., Title matching) or external data
    (e.g., TLD probability) are set to reasonable default/dummy values.
    A production system would require a much more comprehensive implementation.
    """
    features = {}
    try:
        # Use urlsplit for more robust parsing
        parts = urlsplit(url)
        hostname = parts.hostname if parts.hostname else ""
    except Exception:
        # Fallback for invalid URLs
        return pd.DataFrame([dict.fromkeys(numerical_features_for_demo, 0)])

    # --- Feature Extraction (matching numerical_features_for_demo) ---
    features['URLLength'] = len(url)
    features['DomainLength'] = len(parts.netloc)
    
    # IsDomainIP: Check if hostname consists of digits and dots only
    is_ip = 1
    if hostname:
        # Check for IPv6 representation
        if ':' in hostname:
            is_ip = 1
        # Check for IPv4 representation
        elif all(block.isdigit() and 0 <= int(block) <= 255 for block in hostname.split('.')):
             is_ip = 1
        else:
            is_ip = 0
    else:
        is_ip = 0
    features['IsDomainIP'] = is_ip
    
    features['HttpsInURL'] = 1 if parts.scheme == 'https' else 0

    # --- Dummy Features (requiring external data or webpage content) ---
    # These are set to average/neutral values for demonstration purposes.
    features['URLSimilarityIndex'] = 0.5
    features['CharContinuationRate'] = 0.5
    features['URLCharProb'] = -0.5
    features['DomainTitleMatchScore'] = 0.5

    # Ensure all columns are present in the correct order
    final_features = {key: features.get(key, 0) for key in numerical_features_for_demo}

    return pd.DataFrame([final_features])


def predict_url(input_url, vqc_model, scaler_obj):
    """
    Takes a URL string, extracts features, scales them, and predicts
    if it is phishing or legitimate using the trained VQC model.
    """
    print(f"\nAnalyzing URL: {input_url}")
    
    # 1. Extract features
    try:
        feature_vector = extract_features_from_url(input_url)
    except Exception as e:
        return f"Could not parse URL. Error: {e}"

    # 2. Scale features
    scaled_vector = scaler_obj.transform(feature_vector)
    
    # 3. Predict using the VQC model
    try:
        prediction = vqc_model.predict(scaled_vector)
        result = 'Legitimate' if prediction[0] == 1 else 'Phishing'
        return f"Prediction: The URL is likely {result}."
    except (QiskitMachineLearningError, NameError):
        return "Prediction failed. The VQC model may not be trained or available."
    except Exception as e:
        return f"An unexpected error occurred during prediction: {e}"

# --- Example Usage ---
# Note: The VQC model must be successfully trained for this to work.
test_url_legit = "https://www.google.com/search?q=quantum+computing"
test_url_phish = "http://secure-login-update-account.com/paypal.html"
test_url_ip = "http://192.168.1.1/admin"


print(predict_url(test_url_legit, vqc, scaler))
print(predict_url(test_url_phish, vqc, scaler))
print(predict_url(test_url_ip, vqc, scaler))


print("\n--- End of Script ---")