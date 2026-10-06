# VaniShield AI

**Real-time voice anti-spoofing and deepfake detection for live telephone audio.**

[![Status](https://img.shields.io/badge/status-experimental-orange)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](#)
[![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)](#)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)](#license)

> **Project status:** experimental / under active development.

<p align="center">
  <img src="docs/assets/logo.png" alt="VaniShield AI logo" width="280"/>
</p>

---

## Demo Preview

### Screenshots

<p align="center">
  <img src="docs/assets/dashboard-screenshot.png" alt="VaniShield dashboard" width="900"/>
</p>

### Video Demo

[Watch the demo video](docs/assets/Vanishield-AI.mp4)

---

## Overview

**VaniShield AI** is an enterprise-oriented, real-time voice anti-spoofing and deepfake audio detection system designed to protect live communication channels—voice calls, VoIP streams, and contact centers—from synthetic speech and replay attacks.

It combines high-speed statistical heuristics with optional deep learning so legitimate conversations stay low-latency while suspicious or cloned audio is flagged quickly.

### Key capabilities

| Component | Description |
|-----------|-------------|
| **Dual-tier pipeline** | Fast Tier-1 statistical gate, with optional Tier-2 deep verification |
| **Tier-1 analysis** | Mahalanobis-distance detector on forensic acoustic markers (pitch jitter, shimmer, spectral flux, glottal slope, room impulse response, and related cues) |
| **Tier-2 engine** | SSL-AASIST (self-supervised frontend + spectro-temporal graph attention) for high-precision checks on ambiguous streams |
| **Streaming inference** | Ring buffers, sliding windows, resampling; ONNX-oriented path for low-latency deployment |
| **Backend** | FastAPI + Uvicorn with WebSocket audio streaming and Twilio Media Streams support |
| **Ops & UI** | PyTorch training pipeline and a browser dashboard for live monitoring and feature trends |

### Detailed architecture

<p align="center">
  <img src="docs/assets/architecture-diagram.png" alt="Architecture diagram" width="900"/>
</p>

---

## Repository layout

```text
client/                 Browser dashboard
server/                 FastAPI backend and Tier-1 pipeline
ssl_aasist_project/     Optional Tier-2 SSL-AASIST model
sih-voice-detector/     Twilio streaming pipeline components
models/                 Local model artifacts (do not commit)
local_cache/            Training outputs and manifests (do not commit)
fairseq/                Local fairseq dependency (ignored from Git)
train.py                Training entry point
export_onnx.py          PyTorch → ONNX export
test_client.py          Batch streaming / evaluation tests
test_tier1.py           Tier-1 detector tests
```

---

## Requirements

- **Python** 3.10+ (project may target a specific patch release in CI)
- PyTorch and TorchAudio
- FastAPI and Uvicorn
- NumPy, SciPy, scikit-learn, librosa
- Optional: CUDA-capable GPU
- Optional: Twilio account and Media Streams configuration

### Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

---

## Models

Place model files under `models/` (local only; do not commit):

| File | Purpose |
|------|---------|
| `models/master_tier1.pth` | Tier-1 baseline / detector weights |
| `models/SSL_best.pt` | Optional Tier-2 SSL-AASIST checkpoint |
| `models/streaming_model_int8.onnx` | Quantized streaming ONNX model |

---

## Dataset preparation

Organize real and spoof audio by domain:

```text
<data-root>/
  real/
    asvspoof/
    in_the_wild/
    librispeech/
  spoof/
    asvspoof/
    in_the_wild/
    librispeech/
```

Before training, document:

- Dataset download instructions  
- Licenses and attribution  
- Audio format and sample-rate requirements  
- Preprocessing and speaker/session split rules  
- Handling of private or sensitive recordings  

---

## Training

```bash
python train.py --help
python train.py   # add required arguments for your environment
```

Write outputs to a local cache or private artifact store. **Do not commit** audio data, path-heavy manifests, checkpoints, or evaluation recordings.

---

## Exporting the ONNX model

```bash
python export_onnx.py
```

Default export assumes **16 kHz**, mono, window of **1,600 samples**. Confirm shapes and quantization before production use.

---

## Running the backend

```bash
uvicorn server.main:app --host 0.0.0.0 --port 8000
```

Dashboard:

```text
http://localhost:8000/
```

### Deployment checklist

- Public HTTPS / WSS endpoint (required for Twilio)  
- Twilio Media Streams webhook configuration  
- Authentication for WebSocket connections  
- Environment variables for thresholds and feature flags  
- Logging and retention policy for audio and detection results  

---

## Configuration

Use a local `.env` or a secret manager. **Never commit** credentials, API keys, Twilio tokens, private keys, call recordings, or customer data.

Example flags (see backend docs for the full set):

```bash
# Optional: enable Tier-2 on live path
VANISHIELD_LIVE_TIER2=0

# Tier-1 threshold override (if used)
VANISHIELD_TIER1_THRESHOLD=0.56

# Logistic scale for distance → P(spoof)
VANISHIELD_LOGIT_SCALE=4.0
```

---

## Testing

```bash
python -m pytest
python test_tier1.py
python test_client.py
```

Recommended coverage:

- Audio decoding and resampling  
- Streaming buffer boundaries  
- Model input/output shapes  
- Real / spoof threshold behavior  
- WebSocket connect and disconnect handling  
- Twilio event payloads  

### Tier-1 debug (verbose terminal)

For live calls, the backend can print step-by-step Tier-1 math (distance, threshold, \(P(\text{spoof})\), label, confidence). Prefer a dedicated verbose flag or log level in production so high-volume traffic does not flood stdout.

---

## Security and privacy

Voice data may be biometric or personally identifiable. Before deployment:

- Remove secrets from source and Git history  
- Avoid storing raw call audio unless required  
- Define retention, deletion, consent, and access policies  
- Protect WebSocket endpoints  
- Validate upload and stream size/format  
- Treat untrusted model files (e.g. pickle) as unsafe to load  

---

## Contributors

| Name | Role |
|------|------|
| Amit Kumar Singh | Project lead |
| Anoop Kumar | Tier-2 engine and testing |
| Ankit Kumar | Backend / infrastructure |
| Anshul Gupta | Tier-1 engine |
| Vasundhra Rajput | Ingestion model |
| Himanshu Singh | Frontend / dashboard |

---

## Acknowledgments

Thanks to the contributors, reviewers, dataset creators, open-source maintainers, and research communities who support this work—including the teams behind fairseq, SSL-AASIST, PyTorch, FastAPI, ONNX Runtime, and the speech datasets used for evaluation.

---

## License

`<add project license>`

Third-party components (including fairseq and model checkpoints) may use separate licenses. Preserve their license and attribution files before redistribution.

---

## References

1. Voice anti-spoofing / deepfake detection — [arXiv:2510.24852](https://doi.org/10.48550/arxiv.2510.24852)
2. AASIST — [arXiv:2110.01200](https://arxiv.org/abs/2110.01200) · [code](https://github.com/clovaai/aasist)  
   SSL anti-spoofing (wav2vec 2.0) — [Tak et al.](https://github.com/TakHemlata/SSL_Anti-spoofing)
3. [fairseq](https://github.com/facebookresearch/fairseq)
4. [PyTorch docs](https://pytorch.org/docs/)
5. [ONNX Runtime docs](https://onnxruntime.ai/docs/)
6. ASVspoof — [asvspoof.org](https://www.asvspoof.org/) · [2019 DB](https://arxiv.org/abs/1911.01601) · [ASVspoof 5](https://zenodo.org/records/14498691)
7. In-the-Wild deepfakes — [arXiv:2203.16263](https://arxiv.org/abs/2203.16263) · [download](https://deepfake-demo.aisec.fraunhofer.de/in_the_wild)
8. [LibriSpeech](https://www.openslr.org/12)
9. [Web Audio API](https://webaudioapi.com/)

---

<p align="center">
  <sub>Built for safer live voice channels · VaniShield AI</sub>
</p>
