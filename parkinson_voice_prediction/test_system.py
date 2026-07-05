from prediction.predictor import Predictor
predictor = Predictor()

test_files = [
    "dataset/healthy/AH_121A_BD5BA248-E807-4CB9-8B53-47E7FFE5F8E2.wav",
    "dataset/parkinson/sample_0.wav"
]

print("=== Testing DL Voice Model ===")
for f in test_files:
    try:
        pred, prob = predictor.predict_from_audio(f, use_wav2vec=False)
        print(f"{f}: {pred} (Confidence: {prob:.4f})")
    except Exception as e:
        print(f"Error on {f}: {e}")

print("\n=== Testing Classical ML Fallback ===")
for f in test_files:
    try:
        # Simulate fallback by directly calling feature extraction
        from feature_extraction.extract_features import extract_features_from_audio
        import joblib
        selected_features = joblib.load(predictor.selected_features_path)
        features_dict = extract_features_from_audio(f)
        feature_array = [features_dict.get(feat, 0) for feat in selected_features]
        pred, prob = predictor.predict_from_features(feature_array)
        print(f"{f}: {pred} (Confidence: {prob:.4f})")
    except Exception as e:
        print(f"Error on {f}: {e}")
