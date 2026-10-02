"""
VaniShield backend — live path is Tier-1 only (fast).

Spoof decision (corrected):
  1. Tier-1 returns Mahalanobis distance vs real-speech baseline.
  2. P(spoof) = logistic(scale * (distance - threshold))
     — distance at threshold → P=0.5
     — farther above threshold → higher P(spoof)
     — farther below → higher P(real)
  3. is_spoof = P(spoof) >= 0.5
  4. confidence = max(P(spoof), 1 - P(spoof))  # certainty of chosen label

Optional Platt calibrator file:
  models/tier1_platt.json  →  {"a": <float>, "b": <float>}
  with P = sigmoid(a * distance + b)
  If present, overrides the default logistic.
"""

import os
import sys
import json
import asyncio
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from server.pipeline.ring_buffer import RingBuffer
from server.pipeline.vad_filter import VADFilter
from server.pipeline.mahalanobis_detector import Tier1MahalanobisDetector

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
LIVE_USE_TIER2 = os.environ.get("VANISHIELD_LIVE_TIER2", "0").strip() in ("1", "true", "True", "yes")
# Logistic steepness around threshold (higher = sharper transition)
TIER1_LOGIT_SCALE = float(os.environ.get("VANISHIELD_LOGIT_SCALE", "4.0"))
# Optional hard threshold override; set to 16.0 to match your live speech range
_FORCE_THR = os.environ.get("VANISHIELD_TIER1_THRESHOLD", "16.0").strip()
FORCE_TIER1_THRESHOLD = float(_FORCE_THR) if _FORCE_THR else 16.0

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIENT_DIR = os.path.join(BASE_DIR, "client")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="Voice Anti-Spoofing Detector API")

if os.path.exists(CLIENT_DIR):
    app.mount("/client", StaticFiles(directory=CLIENT_DIR), name="client")


@app.get("/")
async def read_root():
    index_file_path = os.path.join(CLIENT_DIR, "index.html")
    if os.path.exists(index_file_path):
        return FileResponse(index_file_path)
    return {
        "message": "Voice Anti-Spoofing Backend is running. Connect via WebSocket at /ws/audio",
        "live_tier2_enabled": LIVE_USE_TIER2,
    }


# ---------------------------------------------------------------------------
# Tier 1
# ---------------------------------------------------------------------------
print(f"[*] Device: {DEVICE}")
print(f"[*] Live Tier-2 escalation: {'ON' if LIVE_USE_TIER2 else 'OFF (Tier-1 only)'}")

tier1_detector = Tier1MahalanobisDetector()
baseline_path = os.path.join(BASE_DIR, "models", "master_tier1.pth")
baseline_loaded = False

if os.path.exists(baseline_path):
    try:
        if baseline_path.endswith(".npy"):
            baseline_data = np.load(baseline_path)
        else:
            import pickle
            with open(baseline_path, "rb") as f:
                baseline_data = pickle.load(f)
        tier1_detector.fit_baseline(baseline_data)
        tier1_detector.calibrate_threshold(percentile=99.0, calibration_features=baseline_data)
        baseline_loaded = True
        print("[*] Tier 1 baseline loaded and calibrated.")
    except Exception as e:
        print(f"[!] Tier 1 baseline load failed: {e}")

TIER1_BASELINE_IS_SYNTHETIC = not baseline_loaded

if not baseline_loaded:
    # Synthetic baseline matching Tier1MahalanobisDetector.extract_8_features() actual output scales
    np.random.seed(42)
    dummy = np.column_stack([
        np.random.uniform(0.00, 0.25, 500),  # 1. jitter
        np.random.uniform(0.20, 1.20, 500),  # 2. shimmer
        np.random.uniform(0.20, 6.00, 500),  # 3. spectral_flux
        np.random.uniform(0.50, 20.0, 500),  # 4. glottal_slope
        np.random.uniform(1.00, 10.0, 500),  # 5. cpp
        np.random.uniform(0.20, 3.00, 500),  # 6. biphase
        np.random.uniform(0.00, 0.08, 500),  # 7. noise_floor
        np.random.uniform(100.0, 900.0, 500) # 8. rir_decay
    ]).astype(np.float32)
    
    tier1_detector.fit_baseline(dummy)
    try:
        tier1_detector.calibrate_threshold(percentile=99.0, calibration_features=dummy)
    except Exception:
        tier1_detector.threshold = 0.56
    print("[*] Tier 1 structured synthetic baseline applied.")

# Threshold: forced override or calibrated
if FORCE_TIER1_THRESHOLD is not None:
    tier1_detector.threshold = float(FORCE_TIER1_THRESHOLD)
    print(f"[*] Tier 1 threshold FORCED to {tier1_detector.threshold}")
else:
    print(f"[*] Tier 1 threshold (calibrated) = {getattr(tier1_detector, 'threshold', None)}")

# Optional Platt calibrator: models/tier1_platt.json → {"a": ..., "b": ...}
# P(spoof) = sigmoid(a * distance + b)
PLATT_A, PLATT_B = None, None
_platt_path = os.path.join(BASE_DIR, "models", "tier1_platt.json")
if os.path.exists(_platt_path):
    try:
        with open(_platt_path, "r") as fh:
            pl = json.load(fh)
        PLATT_A = float(pl["a"])
        PLATT_B = float(pl["b"])
        print(f"[*] Loaded Platt calibrator a={PLATT_A}, b={PLATT_B}")
    except Exception as e:
        print(f"[!] Failed to load Platt calibrator: {e}")


def _sigmoid(x: float) -> float:
    # numerically stable
    if x >= 0:
        z = np.exp(-x)
        return float(1.0 / (1.0 + z))
    z = np.exp(x)
    return float(z / (1.0 + z))


def _p_spoof_from_distance(mah_distance: float, threshold: float) -> float:
    """
    Calibrated-style P(spoof) from Mahalanobis distance.

    Default: logistic centered on threshold so
      distance == threshold → P = 0.5
      distance  > threshold → P > 0.5  (spoof-leaning)
      distance  < threshold → P < 0.5  (real-leaning)

    If Platt (a,b) is loaded: P = sigmoid(a * distance + b)
    """
    d = float(mah_distance)
    if PLATT_A is not None and PLATT_B is not None:
        return _sigmoid(PLATT_A * d + PLATT_B)

    thr = max(float(threshold), 1e-9)
    # scale controls sharpness around the boundary
    return _sigmoid(TIER1_LOGIT_SCALE * (d - thr) / thr)


def _decision_from_p(p_spoof: float) -> tuple:
    """Return (is_spoof, confidence) from P(spoof)."""
    p = float(min(1.0 - 1e-6, max(1e-6, p_spoof)))
    is_spoof = p >= 0.5
    confidence = p if is_spoof else (1.0 - p)
    return is_spoof, float(round(confidence, 4))


# ---------------------------------------------------------------------------
# Tier 2 (lazy)
# ---------------------------------------------------------------------------
tier2_model = None


def _ensure_tier2_loaded():
    global tier2_model
    if tier2_model is not None:
        return tier2_model

    print("[*] Loading Tier 2 SSL-AASIST (lazy)...")
    from ssl_aasist_project.model import SSL_AASIST_Model

    model = SSL_AASIST_Model({"kernel_sizes": [3, 7, 15, 23], "training": False})
    model_path = os.path.join(BASE_DIR, "models", "SSL_best.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "ssl_aasist_project", "models", "SSL_best.pt")

    if os.path.exists(model_path):
        try:
            checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
            state_dict = checkpoint.get(
                "model_state_dict", checkpoint.get("state_dict", checkpoint)
            )
            model.load_state_dict(state_dict, strict=False)
            print(f"[*] Tier 2 weights loaded from {model_path}")
        except Exception as e:
            print(f"[!] Tier 2 weight load failed: {e}")
    else:
        print(f"[!] Tier 2 weights not found at {model_path}")

    model.to(DEVICE)
    model.eval()
    tier2_model = model
    print("[*] Tier 2 ready.")
    return tier2_model


if LIVE_USE_TIER2:
    _ensure_tier2_loaded()
else:
    print("[*] Tier 2 not loaded for live path (set VANISHIELD_LIVE_TIER2=1 to enable).")

_pool = ThreadPoolExecutor(max_workers=1)
print("[*] Pipeline ready.")


def _waveform_for_ui(audio_window: np.ndarray, n_bars: int = 90) -> list:
    if audio_window is None or len(audio_window) == 0:
        return [0.0] * n_bars
    x = np.asarray(audio_window, dtype=np.float32)
    step = max(1, len(x) // n_bars)
    bars = np.abs(x[::step][:n_bars])
    peak = float(bars.max()) if bars.size else 0.0
    if peak > 1e-8:
        bars = bars / peak
    if len(bars) < n_bars:
        bars = np.pad(bars, (0, n_bars - len(bars)))
    return bars.astype(np.float32).tolist()


def _run_tier2(audio_window: np.ndarray) -> tuple:
    model = _ensure_tier2_loaded()
    audio_tensor = torch.from_numpy(np.asarray(audio_window, dtype=np.float32))
    std = float(audio_tensor.std())
    if std > 1e-6:
        audio_tensor = (audio_tensor - audio_tensor.mean()) / std
    waveform_tensor = audio_tensor.unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        output_logits = model(waveform_tensor)
        if output_logits.dim() == 1:
            output_logits = output_logits.unsqueeze(0)
        prediction_idx = int(torch.argmax(output_logits, dim=-1).item())
        probs = torch.softmax(output_logits.float(), dim=-1).squeeze().cpu().tolist()
        if not isinstance(probs, list):
            probs = [float(probs)]
        spoof_prob = float(probs[1]) if len(probs) > 1 else float(probs[0])
    is_fake = bool(prediction_idx == 1)
    return is_fake, spoof_prob


def _extract_live_features(audio_window: np.ndarray, sr: int = 16000, t1_result=None) -> dict:
    x = np.asarray(audio_window, dtype=np.float32).reshape(-1)
    if x.size < 16:
        x = np.pad(x, (0, 16 - x.size))

    rms = float(np.sqrt(np.mean(x * x) + 1e-12))
    frame = max(80, sr // 200)
    hop = max(40, frame // 2)
    n_frames = max(1, (len(x) - frame) // hop + 1)
    energies = np.empty(n_frames, dtype=np.float32)
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame]
        energies[i] = np.sqrt(np.mean(seg * seg) + 1e-12)
    e_mean = float(np.mean(energies) + 1e-12)

    n_fft = int(min(len(x), 2048))
    crop = x[:n_fft] * np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(crop)) + 1e-12
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    ps = spec * spec

    band = (freqs >= 200) & (freqs <= 4000)
    if np.any(band) and np.sum(band) > 4:
        lf = np.log(freqs[band] + 1e-6)
        lp = np.log(ps[band])
        slope = float(np.polyfit(lf, lp, 1)[0])
    else:
        slope = -1.0

    hi = (freqs >= 4000) & (freqs <= min(7000, sr / 2 - 1))
    noise_pow = float(np.mean(ps[hi])) if np.any(hi) else float(np.mean(ps[-8:]))
    noise_floor_db = float(10.0 * np.log10(noise_pow + 1e-12))

    mid = len(x) // 2
    if mid > 8:
        s1 = np.abs(np.fft.rfft(x[:mid] * np.hanning(mid))) + 1e-12
        s2 = np.abs(np.fft.rfft(x[mid:mid * 2] * np.hanning(mid))) + 1e-12
        m = min(len(s1), len(s2))
        flux = float(np.mean(np.abs(s2[:m] - s1[:m]) / (s1[:m] + s2[:m])))
    else:
        flux = 0.0

    shimmer = float(np.mean(np.abs(np.diff(energies))) / e_mean) if len(energies) > 1 else 0.0

    zcrs = []
    for i in range(n_frames):
        seg = x[i * hop : i * hop + frame]
        zcrs.append(np.mean(np.abs(np.diff(np.signbit(seg)).astype(np.float32))))
    zcrs = np.asarray(zcrs, dtype=np.float32)
    pitch_jitter = float(np.std(zcrs) / (np.mean(zcrs) + 1e-6)) if zcrs.size else 0.0

    log_spec = np.log(spec)
    cpp = float(np.max(log_spec) - np.mean(log_spec))

    phase = np.angle(np.fft.rfft(crop))
    if len(phase) > 10:
        dphi = np.diff(phase[: min(64, len(phase))])
        biphase = float(np.abs(np.mean(np.exp(1j * dphi))))
    else:
        biphase = 0.0

    env = energies / (e_mean + 1e-12)
    if len(env) >= 4:
        mid_e = float(np.mean(env[: len(env) // 2]) + 1e-12)
        tail_e = float(np.mean(env[len(env) // 2 :]) + 1e-12)
        rir = float(np.clip(tail_e / mid_e, 0.0, 1.5) * 0.4)
    else:
        rir = 0.0

    features = {
        "pitch_jitter": float(np.clip(pitch_jitter * 2.0, 0.0, 2.0)),
        "shimmer": float(np.clip(shimmer * 3.0, 0.0, 2.0)),
        "spectral_flux": float(np.clip(flux * 2.0, 0.0, 2.0)),
        "glottal": float(np.clip(slope, -20.0, 0.0)),
        "cpp": float(np.clip(cpp, 0.0, 20.0)),
        "biphase": float(np.clip(biphase, 0.0, 1.0)),
        "noise_floor": float(np.clip(noise_floor_db, -120.0, -40.0)),
        "rir": float(np.clip(rir, 0.0, 0.6)),
    }

    if isinstance(t1_result, dict):
        src = t1_result.get("features") or t1_result.get("feature_vector")
        if isinstance(src, dict):
            for k in features:
                if k in src and src[k] is not None:
                    try:
                        features[k] = float(src[k])
                    except Exception:
                        pass
        elif isinstance(src, (list, tuple, np.ndarray)) and len(src) >= 8:
            keys = list(features.keys())
            for i, k in enumerate(keys):
                try:
                    features[k] = float(src[i])
                except Exception:
                    pass

    return {k: round(v, 4) for k, v in features.items()}


def _process_window(audio_window: np.ndarray, vad: VADFilter, use_tier2: bool) -> dict:
    try:
        speech_detected = bool(vad.is_speech(audio_window))
        if not speech_detected:
            return {
                "speech_detected": False,
                "is_fake": False,
                "label": "SILENCE",
                "mahalanobis_distance": 0.0,
                "threshold": float(getattr(tier1_detector, "threshold", 0.56) or 0.56),
                "spoof_probability": 0.0,
                "confidence": 0.50,
                "features": _extract_live_features(audio_window, sr=16000),
                "tier_used": "VAD",
            }

        result_t1 = tier1_detector.evaluate(audio_window, sr=16000)
        if not isinstance(result_t1, dict):
            result_t1 = {"distance": float(result_t1)}

        mah_distance = float(result_t1.get("distance", 0.0))
        threshold = float(
            getattr(tier1_detector, "threshold", None)
            or result_t1.get("threshold")
            or 0.56
        )

        p_spoof = _p_spoof_from_distance(mah_distance, threshold)
        is_fake, conf = _decision_from_p(p_spoof)

        live_features = _extract_live_features(audio_window, sr=16000, t1_result=result_t1)

        if not use_tier2:
            return {
                "speech_detected": True,
                "is_fake": is_fake,
                "label": "SPOOF" if is_fake else "BONAFIDE",
                "mahalanobis_distance": mah_distance,
                "threshold": threshold,
                "spoof_probability": float(round(p_spoof, 4)),
                "confidence": conf,
                "features": live_features,
                "tier_used": "TIER_1_ONLY",
                "baseline_synthetic": TIER1_BASELINE_IS_SYNTHETIC,
            }

        if not is_fake:
            return {
                "speech_detected": True,
                "is_fake": False,
                "label": "BONAFIDE",
                "mahalanobis_distance": mah_distance,
                "threshold": threshold,
                "spoof_probability": float(round(p_spoof, 4)),
                "confidence": conf,
                "features": live_features,
                "tier_used": "TIER_1_FASTPATH",
                "baseline_synthetic": TIER1_BASELINE_IS_SYNTHETIC,
            }

        t2_fake, t2_p = _run_tier2(audio_window)
        p_blend = 0.4 * p_spoof + 0.6 * float(t2_p)
        is_fake2, conf2 = _decision_from_p(p_blend)
        return {
            "speech_detected": True,
            "is_fake": is_fake2,
            "label": "SPOOF" if is_fake2 else "BONAFIDE",
            "mahalanobis_distance": mah_distance,
            "threshold": threshold,
            "spoof_probability": float(round(p_blend, 4)),
            "confidence": conf2,
            "features": live_features,
            "tier_used": "TIER_2_ESCALATED",
            "baseline_synthetic": TIER1_BASELINE_IS_SYNTHETIC,
        }

    except Exception as e:
        print(f"[process_window error] {e}")
        traceback.print_exc()
        return {
            "speech_detected": False,
            "is_fake": False,
            "label": "ERROR",
            "mahalanobis_distance": 0.0,
            "spoof_probability": 0.0,
            "confidence": 0.50,
            "tier_used": "ERROR",
            "error": str(e),
        }


@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"[WEBSOCKET] Client connected (live_tier2={LIVE_USE_TIER2}).")

    try:
        ring_buffer = RingBuffer(capacity_samples=48000, stride_samples=1600)
        vad = VADFilter(threshold=0.3)
    except Exception as e:
        print(f"[WEBSOCKET] Failed to init pipeline: {e}")
        traceback.print_exc()
        try:
            await websocket.send_json(
                {"error": f"pipeline_init_failed: {e}", "is_spoof": False, "confidence": 0.5}
            )
        except Exception:
            pass
        return

    is_processing = False
    call_samples = 0

    try:
        while True:
            try:
                message = await websocket.receive()
            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"[WEBSOCKET] receive failed: {e}")
                break

            if message.get("type") == "websocket.disconnect":
                break

            data = message.get("bytes")
            if data is None:
                continue
            if is_processing:
                continue

            try:
                pcm16 = np.frombuffer(data, dtype=np.int16)
            except Exception:
                continue
            if pcm16.size == 0:
                continue

            float32_data = pcm16.astype(np.float32) / 32768.0
            call_samples += int(float32_data.size)

            try:
                ready = ring_buffer.append(float32_data)
            except Exception as e:
                print(f"[WEBSOCKET] ring_buffer.append error: {e}")
                continue
            if not ready:
                continue

            try:
                audio_window = ring_buffer.get_window()
            except Exception as e:
                print(f"[WEBSOCKET] get_window error: {e}")
                continue

            is_processing = True
            loop = asyncio.get_running_loop()

            try:
                detection = await loop.run_in_executor(
                    _pool, _process_window, audio_window, vad, LIVE_USE_TIER2
                )

                is_spoof = bool(detection.get("is_fake", False))
                mah = float(detection.get("mahalanobis_distance", 0.0))
                thr = float(detection.get("threshold", getattr(tier1_detector, "threshold", 0.56) or 0.56))
                p_spoof = float(detection.get("spoof_probability", 0.0))
                confidence = float(detection.get("confidence", 0.5))
                confidence = float(round(max(0.50, min(0.99, confidence)), 4))

                features = detection.get("features")
                if not isinstance(features, dict) or not features:
                    features = _extract_live_features(audio_window, sr=16000, t1_result=detection)

                duration_s = call_samples / 16000.0
                response = {
                    "waveform": _waveform_for_ui(audio_window, n_bars=90),
                    "features": features,
                    "is_spoof": is_spoof,
                    "confidence": confidence,
                    "spoof_probability": round(p_spoof, 4),
                    "duration": round(duration_s, 1),
                    "tier_used": detection.get("tier_used", "UNKNOWN"),
                    "mahalanobis_distance": round(mah, 4),
                    "tier1_threshold": round(thr, 4),
                    "baseline_synthetic": bool(
                        detection.get("baseline_synthetic", TIER1_BASELINE_IS_SYNTHETIC)
                    ),
                }
                print(
                    f"[T1] dist={mah:.4f} thr={thr:.4f} P(spoof)={p_spoof:.3f} "
                    f"label={'SPOOF' if is_spoof else 'REAL'} conf={confidence:.2f}"
                )
                await websocket.send_json(response)

            except WebSocketDisconnect:
                raise
            except Exception as e:
                print(f"[WS send/process error] {e}")
                traceback.print_exc()
            finally:
                is_processing = False

    except WebSocketDisconnect:
        print("[WEBSOCKET] Client disconnected.")
    except Exception as e:
        print(f"[CRITICAL WEBSOCKET ERROR] {e}")
        traceback.print_exc()
    finally:
        print("[WEBSOCKET] Session ended.")


@app.post("/api/analyze_offline")
async def analyze_offline(file: UploadFile = File(...)):
    import io
    import soundfile as sf

    raw = await file.read()
    try:
        audio, sr = sf.read(io.BytesIO(raw), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != 16000:
            duration = len(audio) / float(sr)
            new_len = int(duration * 16000)
            audio = np.interp(
                np.linspace(0, len(audio) - 1, new_len),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
    except Exception:
        pcm = np.frombuffer(raw, dtype=np.int16)
        audio = pcm.astype(np.float32) / 32768.0

    result_t1 = tier1_detector.evaluate(audio, sr=16000)
    mah = float(result_t1.get("distance", 0.0))
    thr = float(getattr(tier1_detector, "threshold", 0.56) or 0.56)
    p1 = _p_spoof_from_distance(mah, thr)

    try:
        _, p2 = _run_tier2(audio)
        p = 0.4 * p1 + 0.6 * p2
    except Exception as e:
        return {"error": str(e), "mahalanobis": mah, "p_spoof_tier1": p1}

    is_spoof, conf = _decision_from_p(p)
    return {
        "is_spoof": is_spoof,
        "confidence": conf,
        "spoof_probability": float(round(p, 4)),
        "mahalanobis_distance": mah,
        "tier1_threshold": thr,
        "tier_used": "TIER_1+TIER_2_OFFLINE",
    }