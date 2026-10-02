# Import numpy for multi-dimensional array math and matrix inversions
import numpy as np

# Import librosa for audio signal processing and spectral feature extraction
import librosa

# Import scipy signal module to compute autocorrelation and peak detection
import scipy.signal as signal

# Import mahalanobis function to calculate covariance-adjusted statistical distance
from scipy.spatial.distance import mahalanobis

# Import chi2 to calculate theoretical confidence intervals
from scipy.stats import chi2

# Named constant instead of a magic number: distance assigned when feature
# extraction fails or the clip is silent, guaranteeing a REJECT decision.
EXTRACTION_FAILURE_DISTANCE = 999.0

# Named constant for the minimum number of samples needed for a stable 8x8 covariance estimate.
MIN_TRAINING_SAMPLES = 20



class Tier1VoiceDetector:
    """
    Tier 1 One-Class Voice Anomaly Detector.
    Learns only genuine human speech baseline statistics to maximize
    zero-day detection of unseen voice cloning algorithms.
    """

    def __init__(self):
        self.mean_vector = None
        self.inv_covariance = None
        self.threshold = None

    @property
    def is_fitted(self) -> bool:
        return self.mean_vector is not None and self.inv_covariance is not None

    def extract_8_features(self, y: np.ndarray, sr: int = 16000) -> np.ndarray:
        """Extracts 8 physical vocal features with numerical safeguards and log-scaling."""
        try:
            # Guard against silent or extremely short audio samples
            if len(y) < sr * 0.1 or np.max(np.abs(y)) < 1e-4:
                return np.zeros(8, dtype=np.float32)

            # 1. Pitch Jitter (cycle-to-cycle frequency variation)
            autocorr = signal.correlate(y, y, mode='full')[len(y) - 1:]
            peaks, _ = signal.find_peaks(autocorr, distance=sr // 500)
            if len(peaks) > 2:
                jitter = float(np.mean(np.abs(np.diff(np.diff(peaks)))) / (np.mean(np.diff(peaks)) + 1e-6))
            else:
                jitter = 0.0

            # 2. Shimmer (cycle-to-cycle amplitude variation)
            shimmer = float(np.std(np.abs(y)) / (np.mean(np.abs(y)) + 1e-6))

            # 3. Spectral Flux (rate of spectral change across STFT frames)
            stft = np.abs(librosa.stft(y, n_fft=512, hop_length=160))
            flux = float(np.mean(np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))))

            # 4. Glottal Spectral Slope (Low vs High frequency energy drop-off)
            full_spec = np.abs(np.fft.rfft(y))
            freqs = np.fft.rfftfreq(len(y), 1 / sr)
            low_e = np.mean(full_spec[freqs < 1000])
            high_e = np.mean(full_spec[(freqs >= 1000) & (freqs < 8000)])
            glottal_slope = float(low_e / (high_e + 1e-6))

            # 5. Cepstral Peak Prominence (CPP - Sharpness of vocal fold harmonics)
            log_spec = np.log(full_spec + 1e-6)
            cepstrum = np.abs(np.fft.ifft(log_spec))
            valid_que = cepstrum[10:len(cepstrum) // 2]
            if len(valid_que) > 0:
                cpp = float(cepstrum[np.argmax(valid_que) + 10] / (np.mean(cepstrum) + 1e-6))
            else:
                cpp = 0.0

            # 6. Biphase Mark / Phase Continuity across frequencies
            phase_spectrum = np.angle(librosa.stft(y, n_fft=512, hop_length=160))
            biphase = float(np.mean(np.abs(np.diff(phase_spectrum, axis=1))))

            # 7. Noise Floor Estimation (10th percentile lowest magnitude proxy)
            noise_floor = float(np.percentile(np.abs(y), 10))

            # 8. Room Impulse Response (RIR) / Reverberation Decay rate proxy
            centroids = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=512, hop_length=160)[0]
            rir_decay = float(np.std(centroids))

            # Apply log-transformation to skewed features for Gaussian distribution compliance
            feature_vector = np.array([
                np.log1p(jitter),
                np.log1p(shimmer),
                np.log1p(flux),
                np.log1p(glottal_slope),
                cpp,
                biphase,
                np.log1p(noise_floor),
                rir_decay
            ], dtype=np.float32)

            # Sanity check for NaN/Inf corruption
            if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
                return np.zeros(8, dtype=np.float32)

            return feature_vector

        except Exception:
            return np.zeros(8, dtype=np.float32)

    def fit_baseline(self, feature_matrix: np.ndarray):
        """Fits the single-class baseline using only genuine human feature vectors."""
        assert feature_matrix.shape[1] == 8, "Feature matrix must have exactly 8 columns."

        # Clean any invalid rows
        clean_matrix = feature_matrix[~np.isnan(feature_matrix).any(axis=1)]
        clean_matrix = clean_matrix[~np.isinf(clean_matrix).any(axis=1)]

        if len(clean_matrix) < MIN_TRAINING_SAMPLES:
            raise ValueError(
                f"Too few valid training samples ({len(clean_matrix)}) to compute a stable "
                f"8x8 covariance matrix. Need at least {MIN_TRAINING_SAMPLES}."
            )

        # 1. Calculate Mean Vector (mu)
        self.mean_vector = np.mean(clean_matrix, axis=0)

        # 2. Calculate Covariance Matrix (Sigma) with ridge regularization for stability
        cov_matrix = np.cov(clean_matrix, rowvar=False)
        cov_matrix += np.eye(cov_matrix.shape[0]) * 1e-4

        # 3. Compute Pseudo-Inverse (Sigma^-1)
        self.inv_covariance = np.linalg.pinv(cov_matrix)

        # 4. Establish default Chi-Square Threshold (99% confidence interval for 8 degrees of freedom)
        self.threshold = float(np.sqrt(chi2.ppf(0.99, df=8)))

        print(f"[+] Tier 1 One-Class baseline fitted successfully on {len(clean_matrix)} samples.")
        print(f"    - Default Chi-Square Threshold (99% CI): {self.threshold:.2f}")

    def compute_distance(self, y: np.ndarray, sr: int = 16000) -> float:
        """Computes Mahalanobis Distance for a target audio sample against the genuine baseline."""
        # FIX: fail loudly instead of silently returning a fake 0.0 "everything is fine" distance.
        # The previous behavior meant an unfitted detector would report ACCEPT for any input,
        # which is a fail-open state — the wrong default for a security control.
        if not self.is_fitted:
            raise RuntimeError(
                "Detector has not been fitted. Call fit_baseline() before compute_distance()."
            )

        features = self.extract_8_features(y, sr)
        if np.all(features == 0):
            # Treat extraction failure/silent audio as anomalous maximum distance
            return EXTRACTION_FAILURE_DISTANCE

        dist = mahalanobis(features, self.mean_vector, self.inv_covariance)
        return float(dist)

    def evaluate(self, y: np.ndarray, sr: int = 16000) -> dict:
        """Runtime evaluation loop: Accept, Escalate, or Reject based on threshold."""
        # FIX: no more silent 5.0 fallback threshold — compute_distance() now raises
        # before this line is ever reached if the detector isn't fitted, so `self.threshold`
        # is guaranteed to be a real, calibrated value here.
        distance = self.compute_distance(y, sr)
        threshold = self.threshold

        if distance <= threshold:
            status = "ACCEPT (Real Voice)"
        elif distance <= threshold * 1.25:
            status = "ESCALATE (Suspicious / Tier 2 Required)"
        else:
            status = "REJECT (Anomalous / Cloned Voice)"

        return {
            "distance": distance,
            "threshold": threshold,
            "decision": status
        }