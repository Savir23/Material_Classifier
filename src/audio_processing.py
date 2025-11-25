import os
from pathlib import Path
import numpy as np
from scipy.io.wavefile import read, write
AUDIO_DIR = "audio/"
OUTPUT_DIR = "data/processed/"
AMBIENT_DURATION_SEC = 1.5

def ensure_output_dirs():
    Path(OUTPUT_DIR + "ambient/").mkdir(parents=True, exist_ok=True)
    Path(OUTPUT_DIR + "chirp/").mkdir(parents=True, exist_ok=True)

def split_audio_files():
    ensure_output_dirs()

    files = [f for f in os.listdir(AUDIO_DIR) if f.endswith(".wav")]

    print(f"Processing {len(files)} audio files...")

    for filename in files:
        filepath = AUDIO_DIR + filename
        sr, data = read(filepath)

        # Safety: ensure mono
        if len(data.shape) > 1:
            data = data[:, 0]

        # Calculate split point based on sample rate
        split_sample = int(sr * AMBIENT_DURATION_SEC)

        ambient = data[:split_sample]
        chirp = data[split_sample:]

        # Save outputs
        ambient_path = OUTPUT_DIR + f"ambient/{filename.replace('.wav','_ambient.wav')}"
        chirp_path = OUTPUT_DIR + f"chirp/{filename.replace('.wav','_chirp.wav')}"

        write(ambient_path, sr, ambient.astype(np.int16))
        write(chirp_path, sr, chirp.astype(np.int16))

        print(f"Split {filename} → ambient + chirp")

if __name__ == "__main__":
    split_audio_files()