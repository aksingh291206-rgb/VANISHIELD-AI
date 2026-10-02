import pickle
import librosa

# Load your trained detector model
MODEL_PATH = "tier1_detector.pkl"

with open(MODEL_PATH, "rb") as f:
    detector = pickle.load(f)

print("[+] Loaded detector successfully!")

# Path to any audio file you want to test (real or fake)
test_audio_path = "test_sample.wav"  # Change to your file path

try:
    # Load audio at 16kHz (matching training configuration)
    y, sr = librosa.load(test_audio_path, sr=16000)
    
    # Run evaluation
    result = detector.evaluate(y, sr)
    
    print("\n--- Inference Results ---")
    print(f"File Tested: {test_audio_path}")
    print(f"Mahalanobis Distance: {result['distance']:.2f}")
    print(f"Threshold Limit:      {result['threshold']:.2f}")
    print(f"Final Decision:       {result['decision']}")

except Exception as e:
    print(f"[!] Error processing file: {e}")