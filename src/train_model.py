"""Train a simple TensorFlow model using spectral-subtraction features.

Usage:
    python train_model.py

This script:
- Loads features and metadata from the `audio/` directory and `recordings_metadata` (via supabase API),
  mirroring logic from `wavTester.ipynb`.
- Builds a small dense neural network in TensorFlow/Keras.
- Trains the model and saves both a Keras SavedModel and a converted TFLite file.

Notes:
- Designed for compact datasets. For production, add proper dataset checks and callbacks.
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
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

# Config
AMBIENT_DURATION_SEC = 1.5
TARGET_SR = 44100
AUDIO_DIR = Path("../audio/")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Metadata mappings (must match test script)
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

# Load env and supabase metadata
load_dotenv("../.env")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_KEY in ../.env")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("Connected to Supabase")

resp = supabase.table("recordings_metadata").select("*").execute()
df = pd.DataFrame(resp.data)
print(f"Loaded {len(df)} metadata rows")

# Audio utilities

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


# Build dataset from local audio files and metadata
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
        print(f"Missing local file for {decoded}, skipping")
        continue
    feat = extract_features(str(local_path))
    if feat is None:
        print(f"Could not extract features for {decoded}")
        continue
    X_list.append(feat)
    y_list.append(material)
    meta_rows.append(row)

if len(X_list) == 0:
    raise RuntimeError("No feature data found. Ensure WAV files exist and extraction succeeds.")

X = np.stack(X_list)
y = np.array(y_list)
print("Feature matrix shape:", X.shape)

# Attach metadata (size, shape) to features
meta_features = []
for r in meta_rows:
    size_raw = str(r.get("size", "medium")).strip().lower()
    # normalize to small/medium/large tokens
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

# Combine features
# Flatten spectral features and append metadata
n_spectral = X.shape[1]
X_flat = X.reshape((X.shape[0], n_spectral))
X_combined = np.hstack([X_flat, meta_arr])

# Encode labels
le = LabelEncoder()
y_enc = le.fit_transform(y)
num_classes = len(le.classes_)
print("Classes:", list(le.classes_))

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_combined, y_enc, test_size=0.2, random_state=42, stratify=y_enc if len(y_enc)>1 else None)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build a small TF model
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train
epochs = 50 if len(X_train) > 50 else 20
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=8)

# Save scaler, label encoder and model
import joblib
out_dir = Path("../models/")
out_dir.mkdir(parents=True, exist_ok=True)

joblib.dump(scaler, out_dir / "scaler.joblib")
joblib.dump(le, out_dir / "label_encoder.joblib")
model.save(out_dir / "keras_model")
print("Saved Keras model and preprocessing artifacts to ../models/")

# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
try:
    tflite_model = converter.convert()
    (out_dir / "model.tflite").write_bytes(tflite_model)
    print("Saved TFLite model to ../models/model.tflite")
except Exception as e:
    print("TFLite conversion failed:", e)

print("Training complete.")
