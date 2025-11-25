"""Load TFLite model and evaluate on test dataset.

Usage:
    python test_model.py

This script mirrors preprocessing from `train_model.py`, loads the TFLite model
and runs inference on the test split, printing accuracy and per-class counts.
"""
import os
import urllib.parse
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from scipy.fft import fft
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf

# Config
AMBIENT_DURATION_SEC = 1.5
TARGET_SR = 44100
AUDIO_DIR = Path("../audio/")
MODEL_DIR = Path("../models/")

# Mappings
SIZE_MAPPING = {
    'small': 0,
    'medium': 1,
    'large': 2
}

SHAPE_MAPPING = {
    'Flat': 0,
    'Crushed': 1,
    'Cylindrical': 2,
    'Irregular': 3,
    'Spherical': 4
}

# Load env and metadata
load_dotenv("../.env")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in ../.env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
resp = supabase.table("recordings_metadata").select("*").execute()
df = pd.DataFrame(resp.data)

# Utilities

def load_audio(filepath, target_sr=TARGET_SR):
    try:
        y, sr = librosa.load(filepath, sr=target_sr, mono=True, duration=AMBIENT_DURATION_SEC * 2)
        return sr, y
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None


def spectral_subtraction(ambient, chirp):
    n = min(len(ambient), len(chirp))
    ambient = ambient[:n]
    chirp = chirp[:n]
    fft_ambient = np.abs(fft(ambient))
    fft_chirp = np.abs(fft(chirp))
    clean_fft = np.maximum(fft_chirp - fft_ambient, 0)
    return clean_fft[: n // 2]


def extract_features(filepath):
    sr, y = load_audio(filepath)
    if sr is None or y is None:
        return None
    split = int(sr * AMBIENT_DURATION_SEC)
    if len(y) < split * 2:
        return None
    ambient = y[:split]
    chirp = y[split : split * 2]
    return spectral_subtraction(ambient, chirp)

# Build dataset
X_list = []
y_list = []
meta_rows = []

for idx, row in df.iterrows():
    material = row.get("material")
    file_url = row.get("file_url")
    if not file_url or not material:
        continue
    decoded = urllib.parse.unquote(os.path.basename(file_url))
    if not decoded.lower().endswith(".wav"):
        continue
    local_path = AUDIO_DIR / decoded
    if not local_path.exists():
        continue
    feat = extract_features(str(local_path))
    if feat is None:
        continue
    X_list.append(feat)
    y_list.append(material)
    meta_rows.append(row)

if len(X_list) == 0:
    raise RuntimeError("No data found for testing.")

X = np.stack(X_list)
y = np.array(y_list)

# metadata
meta_features = []
for r in meta_rows:
    size_raw = str(r.get("size", "medium")).strip().lower()
    if "small" in size_raw:
        s = 'small'
    elif "large" in size_raw:
        s = 'large'
    else:
        s = 'medium'
    size_num = SIZE_MAPPING.get(s, 1)
    shape_raw = r.get("shape", "Irregular")
    shape_num = SHAPE_MAPPING.get(shape_raw, SHAPE_MAPPING['Irregular'])
    meta_features.append([size_num, shape_num])

meta_arr = np.array(meta_features)
X_flat = X.reshape((X.shape[0], X.shape[1]))
X_combined = np.hstack([X_flat, meta_arr])

# Load preprocessing artifacts
scaler = joblib.load(MODEL_DIR / "scaler.joblib")
le = joblib.load(MODEL_DIR / "label_encoder.joblib")

X_scaled = scaler.transform(X_combined)

y_enc = le.transform(y)

# Load TFLite model
tflite_path = MODEL_DIR / "model.tflite"
if not tflite_path.exists():
    raise RuntimeError(f"TFLite model not found at {tflite_path}")

interpreter = tf.lite.Interpreter(model_path=str(tflite_path))
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Run inference
preds = []
for i in range(X_scaled.shape[0]):
    inp = X_scaled[i:i+1].astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], inp)
    interpreter.invoke()
    out = interpreter.get_tensor(output_details[0]['index'])
    pred = np.argmax(out, axis=1)[0]
    preds.append(pred)

preds = np.array(preds)
accuracy = np.mean(preds == y_enc)
print(f"TFLite model accuracy on dataset: {accuracy:.4f}")

# Per-class counts
for i, cls in enumerate(le.classes_):
    idxs = (y_enc == i)
    if np.sum(idxs) == 0:
        continue
    acc = np.mean(preds[idxs] == i)
    print(f"Class {cls}: {np.sum(idxs)} samples, accuracy: {acc:.3f}")
