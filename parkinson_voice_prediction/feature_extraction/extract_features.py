import os
import numpy as np
import pandas as pd
import librosa
import warnings
warnings.filterwarnings('ignore')

try:
    import parselmouth
    from parselmouth.praat import call
    HAS_PARSELMOUTH = True
except ImportError:
    HAS_PARSELMOUTH = False
    print("WARNING: parselmouth not installed. pip install praat-parselmouth")

def _safe(val, default=0.0):
    if val is None or not np.isfinite(val):
        return default
    return float(val)

def extract_features_from_audio(file_path):
    """
    Extracts comprehensive voice features (MFCC, Jitter, Shimmer, Spectral).
    """
    features = {}
    
    try:
        y, sr = librosa.load(file_path, sr=16000)
        
        # 1. MFCC (13 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)
        for i in range(13):
            features[f'mfcc_{i}'] = mfccs_mean[i]
            
        # 2. Spectral Features
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(y))
        
        # 3. Pitch
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        features['pitch'] = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0.0
        
        # 4. Chroma (12 coefficients)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        for i in range(12):
            features[f'chroma_{i}'] = chroma_mean[i]

        # 5. Praat-based Jitter/Shimmer
        if HAS_PARSELMOUTH:
            snd = parselmouth.Sound(file_path)
            pp = call(snd, "To PointProcess (periodic, cc)", 75, 600)
            features['jitter'] = _safe(call(pp, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3), 0.005)
            features['shimmer'] = _safe(call([snd, pp], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6), 0.03)
        else:
            features['jitter'] = 0.005
            features['shimmer'] = 0.03
            
        # Check against feature_columns.json
        import json
        json_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "feature_columns.json")
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                expected_columns = json.load(f)
            assert len(expected_columns) == len(features), f"Feature length mismatch: expected {len(expected_columns)} but got {len(features)}"

    except Exception as e:
        print(f"Extraction error: {e}")
        return _set_defaults()

    return features

def _set_defaults():
    """Fallback defaults for features."""
    defaults = {}
    for i in range(13): defaults[f'mfcc_{i}'] = 0.0
    defaults['spectral_centroid'] = 0.0
    defaults['zero_crossing_rate'] = 0.0
    defaults['pitch'] = 0.0
    for i in range(12): defaults[f'chroma_{i}'] = 0.0
    defaults['jitter'] = 0.005
    defaults['shimmer'] = 0.03
    return defaults

def process_audio_directory(dataset_path, output_csv="extracted_features.csv"):
    """
    Processes a directory containing subdirectories for classes (e.g., parkinson/, healthy/).
    """
    data = []
    
    for class_folder in os.listdir(dataset_path):
        class_path = os.path.join(dataset_path, class_folder)
        if not os.path.isdir(class_path):
            continue
            
        label = 1 if 'parkinson' in class_folder.lower() else 0
        
        for file in os.listdir(class_path):
            if file.endswith('.wav') or file.endswith('.mp3'):
                file_path = os.path.join(class_path, file)
                print(f"Processing {file_path}")
                features = extract_features_from_audio(file_path)
                if features:
                    features['target'] = label
                    data.append(features)
                    
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"Features saved to {output_csv}")
        return df
    return pd.DataFrame()
