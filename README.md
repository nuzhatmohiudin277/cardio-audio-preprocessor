# Cardiovascular Audio Preprocessor

An advanced data engineering pipeline designed specifically for phonocardiogram (PCG) heart sound analysis and multi-modal AI triage models. 

## Architecture & Features
```mermaid
graph TD
    A[Raw PCG Audio .wav] -->|Librosa Load| B(Acoustic Signal)
    B --> C{Butterworth Bandpass}
    C -->|20Hz - 400Hz Isolation| D[Cleaned Heart Sounds]
    D --> E(MFCC Extraction)
    E -->|13 Audio Features| F[(Pandas DataFrame)]
    F -->|Export| G[cardio_features.csv]
    
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style C fill:#bbf,stroke:#333,stroke-width:2px
    style E fill:#bfb,stroke:#333,stroke-width:2px
    style G fill:#fbb,stroke:#333,stroke-width:2px```
- **Acoustic Filtering:** Applies a 4th-order Butterworth bandpass filter (20Hz - 400Hz) using `scipy.signal` to isolate human heartbeats and eliminate high-frequency room noise.
- **Feature Extraction:** Utilizes `librosa` to compute Mel-Frequency Cepstral Coefficients (MFCCs) tailored for low-frequency acoustic signals.
- **Data Pipeline:** Aggregates statistical variances and means of extracted features, exporting them via `pandas` into an ML-ready CSV format for downstream model training.

## Usage
Run the pipeline via the command line interface:
```bash
python pipeline.py --input ./data --output cardio_features.csv
