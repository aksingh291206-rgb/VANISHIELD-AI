import os
import glob
import random
import pickle
import numpy as np
import librosa
from scipy.stats import chi2
from scipy.spatial.distance import mahalanobis

if __package__:
    from .tier1_voice_detector_v2 import Tier1VoiceDetector
else:
    from tier1_voice_detector_v2 import Tier1VoiceDetector

# --- Config ---
# Using a glob wildcard to find 'train-clean-100' no matter how it's nested
LIBRISPEECH_PATH = r"C:\sih-voice-detector\**\train-clean-100"
MAX_FILES = 1000
VAL_FRACTION = 0.15          # held out from fitting to sanity-check the threshold empirically
RANDOM_SEED = 42
OUTPUT_MODEL_PATH = "tier1_detector.pkl"


def gather_shuffled_files(root: str, max_files: int, seed: int = RANDOM_SEED):
    files = glob.glob(os.path.join(root, "**", "*.flac"), recursive=True)
    if not files:
        raise FileNotFoundError(
            f"No .flac files found under '{root}'. "
            f"Check that LIBRISPEECH_PATH points at an extracted LibriSpeech folder."
        )
    # FIX: shuffle before slicing, so max_files isn't just "whichever speaker folders sort first"
    rng = random.Random(seed)
    rng.shuffle(files)
    return files[:max_files]


def extract_matrix(filepaths, detector):
    rows = []
    for fp in filepaths:
        try:
            y, sr = librosa.load(fp, sr=16000)
        except Exception as e:
            # FIX: skip unreadable/corrupt files instead of crashing the whole run
            print(f"    [!] Skipping unreadable file {fp}: {e}")
            continue
        feats = detector.extract_8_features(y, sr)
        if not np.all(feats == 0):
            rows.append(feats)
    return np.array(rows, dtype=np.float32)


def main():
    print("[*] Starting Tier 1 Training Phase...")

    all_files = gather_shuffled_files(LIBRISPEECH_PATH, MAX_FILES)

    # FIX: hold out a validation slice of GENUINE audio to check the threshold empirically
    split_idx = int(len(all_files) * (1 - VAL_FRACTION))
    train_files, val_files = all_files[:split_idx], all_files[split_idx:]
    print(f"[*] {len(train_files)} files for baseline fitting, {len(val_files)} held out for validation")

    detector = Tier1VoiceDetector()

    train_matrix = extract_matrix(train_files, detector)
    val_matrix = extract_matrix(val_files, detector)

    if len(train_matrix) < 20:
        raise ValueError(
            f"Only {len(train_matrix)} usable training files — need far more "
            f"for a stable 8x8 covariance estimate."
        )

    detector.mean_vector = np.mean(train_matrix, axis=0)
    cov_matrix = np.cov(train_matrix, rowvar=False) + np.eye(8) * 1e-4
    detector.inv_covariance = np.linalg.pinv(cov_matrix)
    detector.threshold = float(np.sqrt(chi2.ppf(0.99, df=8)))

    print("[+] Training complete!")
    print(f"    - Files used for fitting: {len(train_matrix)}")
    print(f"    - Mean Vector Shape: {detector.mean_vector.shape}")
    print(f"    - Inverted Covariance Shape: {detector.inv_covariance.shape}")
    print(f"    - Theoretical Chi-Square Threshold (99%): {detector.threshold:.2f}")

    # FIX: actually check the threshold against held-out genuine audio instead of trusting theory
    if len(val_matrix) > 0:
        val_distances = np.array([
            mahalanobis(row, detector.mean_vector, detector.inv_covariance) for row in val_matrix
        ])
        empirical_fpr = float(np.mean(val_distances > detector.threshold))
        print(f"    - Held-out genuine files tested: {len(val_matrix)}")
        print(f"    - Empirical false-reject rate at this threshold: {empirical_fpr * 100:.2f}% "
              f"(theoretical target: 1.00%)")
        if empirical_fpr > 0.05:
            print("    [!] Empirical FPR is well above the 1% theoretical target — the Gaussian "
                  "assumption may not hold for your data. Consider an empirical percentile "
                  "threshold instead of the chi-square constant.")
    else:
        print("    [!] No held-out files available to sanity-check the threshold.")

    # FIX: persist the fitted detector so you don't have to refit every session
    with open(OUTPUT_MODEL_PATH, "wb") as f:
        pickle.dump(detector, f)
    print(f"[+] Saved fitted detector to {OUTPUT_MODEL_PATH}")

    print("\nReminder: this is a ONE-CLASS baseline fit only on genuine LibriSpeech audio.")
    print("It flags 'statistically unlike LibriSpeech', not 'is a voice clone' specifically —")
    print("for real clone detection, calibrate against labeled genuine + cloned audio (EER).")


if __name__ == "__main__":
    main()