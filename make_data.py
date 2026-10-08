import numpy as np
from scipy.io import wavfile
import os

# Ensure data directory exists
os.makedirs('./data', exist_ok=True)

# Generate 2 seconds of synthetic low-frequency heartbeat sound (50Hz)
sample_rate = 22050
t = np.linspace(0, 2, sample_rate * 2)
synthetic_heartbeat = np.sin(2 * np.pi * 50 * t) * np.exp(-3 * (t % 1))

# Save as a .wav file
wavfile.write('./data/test_heartbeat.wav', sample_rate, synthetic_heartbeat.astype(np.float32))
print("Success: Synthetic heartbeat audio (test_heartbeat.wav) generated in ./data folder!")
