import os
import sys
import time
import numpy as np
from pathlib import Path

try:
    import sounddevice as sd
except ImportError:
    print("[!] Error: 'sounddevice' library is required. Run: pip install sounddevice")
    sys.exit(1)

from server.pipeline.mahalanobis_detector import Tier1MahalanobisDetector

SAMPLE_RATE = 16000
BLOCK_DURATION_SEC = 1.0
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_DURATION_SEC)

def run_live_mic_test():
    print("=" * 65)
    print("      LIVE MICROPHONE TIER-1 MAHALANOBIS DETECTOR (SCALED)")
    print("=" * 65)

    BASE_DIR = Path(__file__).resolve().parent
    baseline_path = os.path.join(BASE_DIR, "models", "master_tier1.pth")

    tier1_detector = Tier1MahalanobisDetector()
    baseline_loaded = False

    if os.path.exists(baseline_path):
        try:
            # Try loading as a PyTorch .pth file first
            import torch
            loaded_data = torch.load(baseline_path, map_location="cpu")
            if isinstance(loaded_data, dict):
                # Extract tensor if stored inside a dictionary checkpoint
                baseline_data = list(loaded_data.values())[0].numpy()
            else:
                baseline_data = loaded_data.numpy()
            
            tier1_detector.fit_baseline(baseline_data)
            tier1_detector.calibrate_threshold(percentile=99.0, calibration_features=baseline_data)
            baseline_loaded = True
            print("[*] Status: SUCCESS - Loaded baseline from PyTorch .pth checkpoint.")
        except Exception:
            try:
                # Fallback to numpy load if it's a raw numpy array saved with .pth extension
                baseline_data = np.load(baseline_path, allow_pickle=True)
                tier1_detector.fit_baseline(baseline_data)
                tier1_detector.calibrate_threshold(percentile=99.0, calibration_features=baseline_data)
                baseline_loaded = True
                print("[*] Status: SUCCESS - Loaded baseline via numpy.")
            except Exception as e:
                print(f"[!] Status: Baseline load warning ({e}). Using realistic synthetic baseline.")
    
    if not baseline_loaded:
        print("[*] Generating calibrated synthetic baseline for live speech...")
        dummy_baseline = np.column_stack([
            np.random.uniform(0.15, 0.35, 300),    # 1. Jitter
            np.random.uniform(0.80, 2.00, 300),    # 2. Shimmer
            np.random.uniform(0.20, 1.00, 300),    # 3. Flux
            np.random.uniform(4.00, 9.00, 300),    # 4. Glottal Slope
            np.random.uniform(2.00, 4.50, 300),    # 5. CPP
            np.random.uniform(2.00, 2.15, 300),    # 6. Biphase
            np.random.uniform(0.001, 0.02, 300),   # 7. Noise Floor
            np.random.uniform(100.0, 1000.0, 300)  # 8. RIR Decay
        ])
        tier1_detector.fit_baseline(dummy_baseline)
        tier1_detector.calibrate_threshold(percentile=99.0, calibration_features=dummy_baseline)
        
        # Widen the threshold slightly so normal speaking variance passes as BONAFIDE
        tier1_detector.threshold *= 1.8
        print(f"[*] Adjusted Threshold (with margin): {tier1_detector.threshold:.4f}")

    print("-" * 65)
    print(f"[*] Microphone active. Speak naturally into your mic. Press Ctrl+C to exit.\n")

    def audio_callback(indata, frames, time_info, status):
        audio_window = indata[:, 0].astype(np.float32)

        # Skip processing if ambient noise is completely silent
        if np.max(np.abs(audio_window)) < 0.01:
            print("[..] Listening (Silence)...", end="\r")
            return

        try:
            # 1. Extract 8 features
            feature_vector = tier1_detector.extract_8_features(audio_window, sr=SAMPLE_RATE)
            
            # 2. Evaluate distance and threshold
            result_t1 = tier1_detector.evaluate(audio_window, sr=SAMPLE_RATE)
            mah_distance = result_t1["distance"]
            threshold = result_t1["threshold"]
            is_anomalous = result_t1["is_anomalous"]
            label = result_t1["label"]

            ratio = mah_distance / max(threshold, 0.001)

            # 3. Confidence score calculation
            if is_anomalous:
                confidence_score = float(round(min(0.99, max(0.51, 0.5 + (ratio - 1.0) * 0.1)), 4))
                decision = "SPOOF / ANOMALY DETECTED (Flagged)"
            else:
                confidence_score = float(round(min(0.99, max(0.51, 1.0 - (mah_distance / threshold) * 0.4)), 4))
                decision = "BONAFIDE (Passed Tier 1)"

            print("\n" + "="*55)
            print(f" [LIVE TIER-1 CALCULATION TRACE]")
            print("-"*55)
            print(f" • Input Samples Captured : {len(audio_window)}")
            print(f" • Extracted Feature Vector:")
            print(f"     1. Jitter      : {feature_vector[0]:.4f}")
            print(f"     2. Shimmer     : {feature_vector[1]:.4f}")
            print(f"     3. Flux        : {feature_vector[2]:.4f}")
            print(f"     4. Glottal     : {feature_vector[3]:.4f}")
            print(f"     5. CPP         : {feature_vector[4]:.4f}")
            print(f"     6. Biphase     : {feature_vector[5]:.4f}")
            print(f"     7. Noise Floor : {feature_vector[6]:.4f}")
            print(f"     8. RIR Decay   : {feature_vector[7]:.4f}")
            print(f" • Mahalanobis Distance   : {mah_distance:.4f}")
            print(f" • Calibration Threshold  : {threshold:.4f}")
            print(f" • Distance Ratio         : {ratio:.2f}x")
            print(f" • Decision Result        : {decision}")
            print(f" • Confidence Score       : {confidence_score * 100:.2f}%")
            print("="*55)
        except Exception as err:
            print(f"[!] Processing Error: {err}")

    try:
        with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, callback=audio_callback):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[*] Stopped by user.")

if __name__ == "__main__":
    run_live_mic_test()