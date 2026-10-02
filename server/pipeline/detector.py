# server/pipeline/detector.py

import torch
import torch.nn as nn
import numpy as np
import librosa
from scipy.signal import find_peaks


class Tier1HeuristicEngine:
    """
    Tier 1 Engine: Fast Acoustic & Physiological Heuristic Analysis.

    Evaluates physiological vocal tract artifacts that AI vocoders struggle to replicate:
    1. Fundamental Frequency (F0) Jitter: Cycle-to-cycle frequency variations caused by 
       vocal cord micro-tremors (human speech has natural instability; AI speech is overly smooth).
    2. Glottal Flow Spectral Slope: Ratio of low-frequency energy (vocal cord closure) to 
       high-frequency energy (over-smoothing artifact detection in neural vocoders).
    3. Micro-Pause Dynamics: Detection of unnaturally continuous speech energy.
    """

    def __init__(self, sample_rate: int = 16000):
        """
        Initializes the heuristic analyzer parameters.

        Args:
            sample_rate (int): Sampling rate of the incoming audio in Hz. Defaults to 16000.
        """
        self.sr = sample_rate

    def compute_pitch_jitter(self, audio: np.ndarray) -> float:
        """
        Computes the relative pitch jitter (Fundamental Frequency perturbation).

        Jitter = (Mean Absolute Difference of Consecutive Pitch Periods) / (Mean Pitch Period)

        Human vocal cords exhibit small involuntary muscle contractions, resulting in subtle
        pitch period differences. Synthetic voice models (e.g., ElevenLabs, VALL-E) produce
        pitch contours that are unnaturally stable (ultra-low jitter).

        Args:
            audio (np.ndarray): 1D float32 audio signal normalized in [-1.0, 1.0].

        Returns:
            float: Relative pitch jitter ratio (0.0 if unvoiced/insufficient frames).
        """
        # Estimate fundamental pitch contour using Probabilistic YIN (pYIN)
        # Pitch range restricted to C2 (65.4 Hz) - C7 (2093.0 Hz) for human vocal range
        f0, voiced_flag, _ = librosa.pyin(
            audio, 
            fmin=librosa.note_to_hz('C2'), 
            fmax=librosa.note_to_hz('C7'), 
            sr=self.sr
        )
        
        # Filter out NaN values representing unvoiced frames or silence
        f0_clean = f0[~np.isnan(f0)]
        
        # Require a minimum of 5 voiced frames to calculate pitch perturbation
        if len(f0_clean) < 5:
            return 0.0  # Insufficient voiced audio

        # Convert pitch frequencies (Hz) into fundamental periods (seconds): T = 1 / F0
        periods = 1.0 / f0_clean
        
        # Calculate absolute difference between adjacent pitch periods: |T_{i+1} - T_i|
        period_diffs = np.abs(np.diff(periods))
        
        # Compute relative jitter ratio
        jitter = np.mean(period_diffs) / np.mean(periods)
        return float(jitter)

    def compute_glottal_spectral_slope(self, audio: np.ndarray) -> float:
        """
        Evaluates the glottal airflow dynamics by calculating the energy ratio between 
        low-frequency fundamental bands and high-frequency harmonic bands.

        Glottal Spectral Slope = (Energy in [0, 1000 Hz]) / (Energy in [1000 Hz, 8000 Hz])

        Neural vocoders often over-smooth or introduce phase artifacts above 1 kHz, 
        altering the expected spectral tilt of human vocal fold closures.

        Args:
            audio (np.ndarray): 1D float32 audio signal.

        Returns:
            float: Low-to-high frequency energy ratio.
        """
        # Compute Short-Time Fourier Transform (STFT) magnitude spectrum
        stft_magnitude = np.abs(librosa.stft(audio))
        
        # Get frequency bin centers corresponding to the STFT rows
        freq_bins = librosa.fft_frequencies(sr=self.sr)
        
        # Sum energy in low frequency band (< 1000 Hz) representing vocal fold pulses
        low_band_energy = np.mean(stft_magnitude[freq_bins < 1000])
        
        # Sum energy in high frequency band (>= 1000 Hz) representing upper harmonics/friction
        # Added epsilon (+1e-6) to prevent division by zero in dead silence
        high_band_energy = np.mean(stft_magnitude[freq_bins >= 1000]) + 1e-6
        
        spectral_slope = low_band_energy / high_band_energy
        return float(spectral_slope)

    def analyze(self, audio: np.ndarray) -> dict:
        """
        Executes all Tier 1 heuristic feature extractions and returns an initial 
        spoof probability score.

        Args:
            audio (np.ndarray): 1D float32 audio signal.

        Returns:
            dict: Raw heuristic metrics and computed Tier 1 spoof probability.
        """
        jitter = self.compute_pitch_jitter(audio)
        slope = self.compute_glottal_spectral_slope(audio)

        # Anomaly Scoring Strategy:
        # Extremely low jitter (< 0.008) is a strong signal of synthetic/robotic pitch perfection.
        if jitter == 0.0:
            pitch_anomaly = 0.5  # Neutral score if pitch detection could not resolve voiced frames
        elif jitter < 0.008:
            pitch_anomaly = 1.0  # Synthetic pitch perfection flag
        else:
            # Scaled decay score: As jitter approaches natural human variation (~0.03), anomaly drops to 0
            pitch_anomaly = max(0.0, 1.0 - (jitter / 0.03))
        
        # Spectral slope check: Extremely low slope (< 1.5) indicates missing glottal resonance
        spectral_anomaly = 1.0 if slope < 1.5 else 0.0

        # Weighted combination for Tier 1 spoof probability (65% Pitch Jitter, 35% Spectral Tilt)
        t1_spoof_prob = (0.65 * pitch_anomaly) + (0.35 * spectral_anomaly)
        
        return {
            "t1_spoof_prob": round(t1_spoof_prob, 4),
            "jitter": round(jitter, 6),
            "glottal_slope": round(slope, 4)
        }


class Tier2DeepNeuralEngine(nn.Module):
    """
    Tier 2 Engine: Deep Anti-Spoofing Neural Architecture (AASIST & RawNet3 Inspired).

    Accepts raw audio waveforms directly without manual STFT preprocessing, 
    preserving phase and high-frequency micro-anomalies:
    1. Sinc-Convolution Layer: Extracts raw waveform temporal representations across learned sub-bands.
    2. Spectral-Temporal Graph Aggregator: Simulates Graph Neural Network (GNN) node aggregation 
       to model non-local relationships across spectral bins and time frames simultaneously.
    3. Binary Classification Head: Maps graph embeddings to a spoof probability scalar [0.0, 1.0].
    """

    def __init__(self):
        super().__init__()
        
        # Raw waveform front-end feature extractor (RawNet3 style Sinc-conv inspired layer)
        # Input shape:  (Batch, 1, 48000) for 3-second 16kHz audio
        # Output shape: (Batch, 64, 2999) after strided convolution and max pooling
        self.raw_conv = nn.Sequential(
            # Stage 1: Capture fine temporal micro-structures using wide kernel
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=128, stride=4, padding=64),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(kernel_size=4),
            
            # Stage 2: Aggregate high-level acoustic representations
            nn.Conv1d(in_channels=32, out_channels=64, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        
        # Graph aggregation representation (Simulating AASIST Graph Neural Network topology)
        # Evaluates joint spectro-temporal relationships across distant time-frequency nodes
        self.gnn_aggregate = nn.Sequential(
            nn.AdaptiveAvgPool1d(16),  # Standardize temporal dimension to 16 graph nodes
            nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)     # Global graph pooling -> shape: (Batch, 128, 1)
        )
        
        # Classification MLP
        self.classifier = nn.Sequential(
            nn.Linear(in_features=128, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=1),
            nn.Sigmoid()  # Output bound between 0.0 (Human) and 1.0 (Synthetic Spoof)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through deep anti-spoofing pipeline.

        Args:
            x (torch.Tensor): Raw waveform tensor of shape (Batch, 1, Num_Samples).

        Returns:
            torch.Tensor: Spoof probability tensor of shape (Batch, 1).
        """
        raw_features = self.raw_conv(x)
        gnn_features = self.gnn_aggregate(raw_features).squeeze(-1)
        spoof_probability = self.classifier(gnn_features)
        return spoof_probability


class DoubleTierDetectorEngine:
    """
    Orchestrates the two-tier hybrid inference pipeline:
    
    Processing Flow:
    1. Every speech-detected buffer is evaluated by Tier 1 (Ultra-fast physiological heuristics).
    2. If Tier 1 confidence is definitive (outside ambiguous zone [low_threshold, high_threshold]), 
       return Tier 1 result immediately to save compute latency.
    3. If Tier 1 result is ambiguous, escalate to Tier 2 (Deep RawNet3/AASIST Neural Model) 
       for deep structural verification.
    """

    def __init__(self, low_threshold: float = 0.30, high_threshold: float = 0.70):
        """
        Initializes both detection tiers and configures escalation boundaries.

        Args:
            low_threshold (float): Lower bound for ambiguity. Scores below this are instantly BONAFIDE.
            high_threshold (float): Upper bound for ambiguity. Scores above this are instantly SPOOF.
        """
        # Auto-select hardware acceleration (CUDA GPU if available, fallback to CPU)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Instantiate Tier 1 Heuristic Engine
        self.tier1_heuristics = Tier1HeuristicEngine(sample_rate=16000)
        
        # Instantiate Tier 2 Deep Neural Engine and move weights to target computing device
        self.tier2_nn = Tier2DeepNeuralEngine().to(self.device)
        self.tier2_nn.eval()  # Set PyTorch model to evaluation mode (disables dropout/batch norm training)

        # Ambiguity bounds that control whether Tier 2 escalation triggers
        self.low_thresh = low_threshold
        self.high_thresh = high_threshold

    def analyze_buffer(self, audio_chunk: np.ndarray) -> dict:
        """
        Analyzes a 3-second audio window through the dual-tier pipeline.

        Args:
            audio_chunk (np.ndarray): 1D float32 audio sample array normalized in [-1.0, 1.0].

        Returns:
            dict: Structured classification results including probabilities, decision labels,
                  tier execution metadata, and extracted physiological metrics.
        """
        # Handle empty or invalid audio input
        if len(audio_chunk) == 0:
            return {
                "spoof_probability": 0.0,
                "human_probability": 1.0,
                "label": "UNCERTAIN",
                "tier_used": "NONE"
            }

        # ------------------- TIER 1: Fast Heuristic Analysis -------------------
        t1_results = self.tier1_heuristics.analyze(audio_chunk)
        t1_prob = t1_results["t1_spoof_prob"]

        # Check if Tier 1 score falls into the ambiguous region requiring deep neural verification
        is_ambiguous = self.low_thresh <= t1_prob <= self.high_thresh

        if not is_ambiguous:
            # Fast Path: Tier 1 decision is clear; bypass Tier 2 GPU computation
            final_prob = t1_prob
            tier_used = "TIER_1_HEURISTIC_GLOTTAL"
            t2_prob = None
        else:
            # ---------------- TIER 2: Deep Anti-Spoofing Escalation ----------------
            # Reshape 1D numpy array into 3D PyTorch Tensor: (Batch=1, Channels=1, Samples)
            tensor_input = torch.from_numpy(audio_chunk).float().unsqueeze(0).unsqueeze(0).to(self.device)
            
            # Disable gradient computation for faster inference and lower RAM usage
            with torch.no_grad():
                t2_prob = float(self.tier2_nn(tensor_input).item())

            # Weighted ensemble combination prioritizing Tier 2 deep representations (30% Tier 1 + 70% Tier 2)
            final_prob = (0.30 * t1_prob) + (0.70 * t2_prob)
            tier_used = "TIER_2_AASIST_RAWNET3_GNN"

        # Binary decision mapping based on 0.50 probability boundary
        label = "SYNTHETIC_SPOOF" if final_prob >= 0.50 else "BONAFIDE_HUMAN"

        # Return comprehensive diagnostic payload
        return {
            "spoof_probability": round(final_prob, 4),
            "human_probability": round(1.0 - final_prob, 4),
            "label": label,
            "tier_used": tier_used,
            "tier1_metrics": {
                "pitch_jitter": t1_results["jitter"],
                "glottal_spectral_slope": t1_results["glottal_slope"],
            },
            "tier2_score": round(t2_prob, 4) if t2_prob is not None else None,
        }

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import json

# Initialize FastAPI application instance (Uvicorn looks for 'app')
app = FastAPI(title="Dual-Tier Voice Anti-Spoofing Pipeline", version="1.0.0")

# Initialize the dual-tier engine globally to keep weights loaded in memory
detector_engine = DoubleTierDetectorEngine()

@app.get("/")
def root():
    return {"status": "online", "pipeline": "Dual-Tier Voice Anti-Spoofing Active"}

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time streaming audio inference.
    Receives raw PCM/float audio chunks from the client, passes them through
    the DoubleTierDetectorEngine, and streams back detection results.
    """
    await websocket.accept()
    try:
        while True:
            # Receive audio data from client (expecting raw bytes or JSON payload)
            message = await websocket.receive()
            
            if "bytes" in message:
                raw_bytes = message["bytes"]
                # Convert incoming raw PCM bytes (assuming float32 or int16) to numpy array
                audio_chunk = np.frombuffer(raw_bytes, dtype=np.float32)
            elif "text" in message:
                data = json.loads(message["text"])
                audio_chunk = np.array(data.get("audio", []), dtype=np.float32)
            else:
                continue

            # Run through dual-tier detector pipeline
            result = detector_engine.analyze_buffer(audio_chunk)

            # Send back analysis telemetry to client
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("Client disconnected from audio stream.")