import librosa
import pandas as pd
import numpy as np
import scipy.signal as signal
import os
import argparse

def apply_bandpass_filter(audio_data, sr, lowcut=20.0, highcut=400.0):
    """
    Apply a Butterworth bandpass filter to isolate human heart sounds (PCG).
    Filters out background room noise and high-frequency artifacts.
    """
    nyquist = 0.5 * sr
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # 4th order Butterworth filter
    b, a = signal.butter(4, [low, high], btype='band')
    filtered_audio = signal.lfilter(b, a, audio_data)
    return filtered_audio

def extract_cardio_features(directory_path, output_csv):
    """
    Scans directory for raw audio, applies bandpass filtering, 
    extracts MFCCs, and saves to an ML-ready CSV format.
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created. Please add raw .wav files.")
        return

    features = []
    print(f"Scanning '{directory_path}' for audio files...")
    
    for file_name in os.listdir(directory_path):
        if file_name.endswith('.wav'):
            file_path = os.path.join(directory_path, file_name)
            
            try:
                # Load audio at native sample rate
                y, sr = librosa.load(file_path, sr=None)
                
                # Preprocess: Clean acoustic noise
                clean_y = apply_bandpass_filter(y, sr)
                
                # Extract robust features (13 MFCCs is standard for this frequency range)
                mfcc = librosa.feature.mfcc(y=clean_y, sr=sr, n_mfcc=13)
                
                features.append({
                    'filename': file_name, 
                    'mfcc_mean': np.mean(mfcc),
                    'mfcc_variance': np.var(mfcc)
                })
                print(f"Successfully processed: {file_name}")
                
            except Exception as e:
                print(f"Error processing {file_name}: {e}")
    
    if features:
        df = pd.DataFrame(features)
        df.to_csv(output_csv, index=False)
        print(f"Pipeline complete! Features exported to {output_csv}")
    else:
        print("No .wav files found to process.")

if __name__ == "__main__":
    # Setup argument parser for CLI usage
    parser = argparse.ArgumentParser(description="Cardiovascular Audio Preprocessor")
    parser.add_argument('--input', type=str, default='./data', help='Input directory for raw .wav files')
    parser.add_argument('--output', type=str, default='cardio_features.csv', help='Output CSV file name')
    
    args = parser.parse_args()
    extract_cardio_features(args.input, args.output)
