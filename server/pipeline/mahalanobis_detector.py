"""KD-tree based Tier 1 detector used by the evaluation scripts.

Run a local smoke test from the repository root with:

    python -m server.pipeline.mahalanobis_detector --audio test_sample.wav
"""

from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import scipy.signal as signal
from scipy.spatial import KDTree


FEATURE_COUNT = 8
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_NEIGHBORS = 3


class Tier1MahalanobisDetector:
    """Score audio against a baseline of genuine-voice feature vectors."""

    def __init__(self, neighbors: int = DEFAULT_NEIGHBORS):
        if neighbors < 1:
            raise ValueError("neighbors must be at least 1")
        self.neighbors = neighbors
        self.scaled_baseline: np.ndarray | None = None
        self.scaler_mean: np.ndarray | None = None
        self.scaler_std: np.ndarray | None = None
        self.kdtree: KDTree | None = None
        self.threshold: float | None = None

    @property
    def is_fitted(self) -> bool:
        return self.kdtree is not None

    def fit_baseline(self, feature_matrix: np.ndarray) -> None:
        """Fit normalization and nearest-neighbor lookup from an ``(N, 8)`` matrix."""
        matrix = np.asarray(feature_matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
            raise ValueError(
                f"Baseline must have shape (N, {FEATURE_COUNT}); got {matrix.shape}."
            )
        if len(matrix) < self.neighbors:
            raise ValueError("Baseline contains fewer rows than the neighbor count.")
        if not np.isfinite(matrix).all():
            raise ValueError("Baseline contains NaN or infinite feature values.")

        self.scaler_mean = np.mean(matrix, axis=0)
        self.scaler_std = np.std(matrix, axis=0) + 1e-6
        self.scaled_baseline = (matrix - self.scaler_mean) / self.scaler_std
        self.kdtree = KDTree(self.scaled_baseline)

    def calibrate_threshold(
        self,
        percentile: float = 99.0,
        calibration_features: np.ndarray | None = None,
    ) -> float:
        """Calibrate from held-out genuine features when they are available."""
        if not self.is_fitted:
            raise RuntimeError("Baseline is not fitted. Call fit_baseline() first.")
        if not 0.0 < percentile < 100.0:
            raise ValueError("percentile must be between 0 and 100.")
        assert self.kdtree is not None

        if calibration_features is not None:
            matrix = np.asarray(calibration_features, dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[1] != FEATURE_COUNT:
                raise ValueError(
                    f"Calibration features must have shape (N, {FEATURE_COUNT}); "
                    f"got {matrix.shape}."
                )
            if len(matrix) < self.neighbors:
                raise ValueError("Calibration contains fewer rows than the neighbor count.")
            if not np.isfinite(matrix).all():
                raise ValueError("Calibration contains NaN or infinite feature values.")
            assert self.scaler_mean is not None
            assert self.scaler_std is not None
            query_points = (matrix - self.scaler_mean) / self.scaler_std
            distances, _ = self.kdtree.query(query_points, k=self.neighbors)
            calibration_distances = np.mean(np.atleast_2d(distances), axis=1)
        else:
            assert self.scaled_baseline is not None
            query_neighbors = min(self.neighbors + 1, len(self.scaled_baseline))
            distances, _ = self.kdtree.query(self.scaled_baseline, k=query_neighbors)
            if query_neighbors == 1:
                calibration_distances = np.atleast_1d(distances)
            else:
                calibration_distances = np.mean(distances[:, 1:], axis=1)
        self.threshold = float(np.percentile(calibration_distances, percentile))
        return self.threshold

    def extract_8_features(
        self, 
        audio: np.ndarray, 
        sr: int = DEFAULT_SAMPLE_RATE, 
        sample_rate: int | None = None
    ) -> np.ndarray:
        active_sr = sample_rate if sample_rate is not None else sr

        y = np.asarray(audio, dtype=np.float32).squeeze()
        if y.ndim != 1 or y.size < 2:
            raise ValueError("Audio must be a non-empty one-dimensional array.")
        if active_sr <= 0:
            raise ValueError("sample_rate must be positive.")

        # 1. Jitter (Autocorrelation peak variance)
        autocorrelation = signal.correlate(y, y, mode="full")[len(y) - 1:]
        peaks, _ = signal.find_peaks(autocorrelation, distance=max(1, active_sr // 500))
        if len(peaks) > 2:
            peak_deltas = np.diff(peaks)
            jitter = float(np.mean(np.abs(np.diff(peak_deltas))) / (np.mean(peak_deltas) + 1e-6))
        else:
            jitter = 0.0

        # 2. Shimmer
        absolute_audio = np.abs(y)
        shimmer = float(np.std(absolute_audio) / (np.mean(absolute_audio) + 1e-6))

        # 3. Spectral Flux (Using STFT frames)
        stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
        flux = float(np.mean(np.sqrt(np.sum(np.diff(stft, axis=1) ** 2, axis=0))))

        # 4. Glottal Slope (Averaged across STFT frames instead of global FFT)
        freqs = librosa.fft_frequencies(sr=active_sr, n_fft=2048)
        low_mask = freqs < 1000
        high_mask = (freqs >= 1000) & (freqs < 8000)
        low_energy = np.mean(stft[low_mask, :], axis=0)
        high_energy = np.mean(stft[high_mask, :], axis=0)
        glottal_slope = float(np.mean(low_energy / (high_energy + 1e-6)))

        # 5. CPP (Frame-averaged log-spectrum cepstrum)
        log_stft = np.log(stft + 1e-6)
        cepstrums = np.abs(np.fft.ifft(log_stft, axis=0))
        # Look within human F0 quefrency bounds (e.g., 60Hz to 330Hz)
        min_q = int(active_sr / 330)
        max_q = int(active_sr / 60)
        if min_q < 1: min_q = 1
        if max_q >= cepstrums.shape[0]: max_q = cepstrums.shape[0] - 1
        
        if min_q < max_q:
            q_region = cepstrums[min_q:max_q, :]
            peak_vals = np.max(q_region, axis=0)
            mean_vals = np.mean(cepstrums, axis=0)
            cpp = float(np.mean(peak_vals / (mean_vals + 1e-6)))
        else:
            cpp = 0.0

        # 6. Biphase
        phase_spectrum = np.angle(librosa.stft(y, n_fft=2048, hop_length=512))
        biphase = float(np.mean(np.abs(np.diff(phase_spectrum, axis=1))))

        # 7. Noise Floor
        normalized_audio = y / (np.max(absolute_audio) + 1e-6)
        noise_floor = float(np.percentile(np.abs(normalized_audio), 10))

        # 8. RIR Decay
        centroids = librosa.feature.spectral_centroid(y=y, sr=active_sr, n_fft=2048, hop_length=512)[0]
        rir_decay = float(np.std(centroids))

        features = np.array(
            [jitter, shimmer, flux, glottal_slope, cpp, biphase, noise_floor, rir_decay],
            dtype=np.float32,
        )
        if not np.isfinite(features).all():
            raise ValueError("Feature extraction produced NaN or infinite values.")
        return features

    def extract_features_from_audio(
        self, audio: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE, sample_rate: int | None = None
    ) -> np.ndarray:
        """Compatibility API returning one baseline row for a complete audio clip."""
        return self.extract_8_features(audio, sr=sr, sample_rate=sample_rate).reshape(1, FEATURE_COUNT)

    def compute_distance(
        self, audio: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE, sample_rate: int | None = None
    ) -> float:
        """Return the mean standardized distance to the nearest baseline voices."""
        if not self.is_fitted:
            raise RuntimeError("Baseline is not fitted. Call fit_baseline() first.")
        assert self.scaler_mean is not None
        assert self.scaler_std is not None
        assert self.kdtree is not None

        features = self.extract_8_features(audio, sr=sr, sample_rate=sample_rate)
        scaled_features = (features - self.scaler_mean) / self.scaler_std
        distances, _ = self.kdtree.query(scaled_features, k=self.neighbors)
        return float(np.mean(np.atleast_1d(distances)))

    def evaluate(
        self, audio: np.ndarray, sr: int = DEFAULT_SAMPLE_RATE, sample_rate: int | None = None
    ) -> dict[str, float | bool | str]:
        """Return the distance and calibrated real/anomalous decision."""
        if self.threshold is None:
            raise RuntimeError("Threshold is not calibrated. Call calibrate_threshold() first.")
        distance = self.compute_distance(audio, sr=sr, sample_rate=sample_rate)
        is_anomalous = distance > self.threshold
        return {
            "distance": distance,
            "threshold": self.threshold,
            "is_anomalous": is_anomalous,
            "label": "SPOOF" if is_anomalous else "BONAFIDE",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, type=Path, help="Audio file to score")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=Path("models/master_tier1.pth"),
        help="Path to an 8-column .npy baseline",
    )
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    args = parser.parse_args()

    if not args.audio.exists():
        parser.error(f"Audio file not found: {args.audio}")
    if not args.baseline.exists():
        parser.error(f"Baseline file not found: {args.baseline}")

    audio, sample_rate = librosa.load(args.audio, sr=args.sample_rate, mono=True)
    detector = Tier1MahalanobisDetector()
    detector.fit_baseline(np.load(args.baseline))
    threshold = detector.calibrate_threshold()
    result = detector.evaluate(audio, sample_rate=sample_rate)
    print(f"audio={args.audio}")
    print(f"samples={len(audio)} sample_rate={sample_rate}")
    print(f"distance={result['distance']:.4f}")
    print(f"threshold={threshold:.4f}")
    print(f"prediction={result['label']}")


if __name__ == "__main__":
    main()