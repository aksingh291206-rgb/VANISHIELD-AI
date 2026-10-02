"""
Live-call voice spoof detector training - hardened, multi-domain version.

Actual dataset composition (from inspection):
  Real:
    - real/asvspoof/     :  2,580 files   (controlled clean)
    - real/in_the_wild/  : 19,963 files   (podcasts, YouTube, real-world)
    - real/librispeech/  : 74,344 files   (clean studio audiobook)
    - Total Real         : 96,887 files
  Spoof:
    - spoof/asvspoof/    : 22,800 files   (classic TTS/VC artifacts)
    - spoof/in_the_wild/ : 11,816 files   (modern deepfakes)
    - spoof/librispeech/ :      0 files   (none)
    - Total Spoof        : 34,616 files

Anti-bias measures:
  1. Domain-aware discovery of sub-folders.
  2. Strict per-domain train/val splits (no leakage across domains), with an
     optional speaker/session-aware grouped split to avoid the same
     speaker appearing in both train and val.
  3. Domain-balanced sampling every block via a cyclic, without-replacement
     sampler per domain: every training block gets EXACTLY the requested
     size (no silent shortfall), and every file in a domain's pool is used
     once before any file repeats (verifiable "coverage", not blind
     resampling).
  4. Exact 1:1 Real:Spoof block sizing when block_real == block_spoof.
  5. Identical telephony augmentation applied to every domain.
  6. Peak normalisation + reproducible random crops for both train and val.
  7. Librispeech Real is deliberately capped (default weight 0.25) so the
     74k clean files cannot dominate learning.

Robustness / correctness hardening:
  - Domain-balanced allocation is exact (largest-remainder method).
  - Validation reports BOTH a clean metric and a deterministic
    telephone-simulated metric (bandpass + mu-law, no randomness).
  - Per-domain metric breakdown every epoch (surfaces environment shortcuts).
  - Optional final evaluation against a completely separate external
    holdout set (--external_real_dir / --external_spoof_dir).
  - Optional speaker/session-grouped splitting (--speaker_group_regex).
  - Defensive data loading: skips empty/non-1D/NaN-Inf arrays.
  - Non-finite (NaN/Inf) loss detection during training.
  - Full argument validation with clear error messages.
  - Deterministic given --seed.
  - Run manifest written to cache_dir/run_manifest.json.
  - Graceful Ctrl-C handling.
  - tqdm is optional.

Remember: even with all of the above, final evaluation should still use a
completely external telephony / modern-deepfake test set that was never
touched during development.
"""

import argparse
import glob
import json
import os
import random
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.metrics import roc_auc_score, roc_curve

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        return iterable

try:
    from scipy.signal import butter, sosfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("Warning: scipy not found - telephone band-limit will be skipped. "
          "Install with: pip install scipy")


# ===========================================================
# Reproducibility
# ===========================================================
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ===========================================================
# Metrics
# ===========================================================
def compute_eer(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """EER with linear interpolation between the ROC points that straddle fpr==fnr."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1.0 - tpr
    abs_diff = np.abs(fpr - fnr)
    idx = int(np.nanargmin(abs_diff))
    if idx == 0 or idx == len(fpr) - 1 or abs_diff[idx] < 1e-6:
        eer = (fpr[idx] + fnr[idx]) / 2.0
        thr = thresholds[idx]
        return float(eer), float(thr)
    if fpr[idx] > fnr[idx]:
        i1, i2 = idx - 1, idx
    else:
        i1, i2 = idx, idx + 1
    denom = (fnr[i1] - fpr[i1]) - (fnr[i2] - fpr[i2])
    if abs(denom) < 1e-12:
        eer = (fpr[idx] + fnr[idx]) / 2.0
        thr = thresholds[idx]
    else:
        t = (fnr[i1] - fpr[i1]) / denom
        eer = fpr[i1] + t * (fpr[i2] - fpr[i1])
        thr = thresholds[i1] + t * (thresholds[i2] - thresholds[i1])
    return float(eer), float(thr)


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    """Run a model over a loader and return raw scores plus EER/AUC."""
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for waveforms, y in loader:
            logits = model(waveforms.to(device))
            s = torch.sigmoid(logits).cpu().numpy()
            scores.append(s)
            labels.append(y.numpy())
    y_true = np.concatenate(labels) if labels else np.zeros(0, dtype=np.float32)
    y_score = np.concatenate(scores) if scores else np.zeros(0, dtype=np.float32)
    if len(set(y_true.tolist())) < 2:
        raise RuntimeError(
            "Evaluation set has only one class present - cannot compute EER/AUC. "
            "Check val_target_real/val_target_spoof and domain availability."
        )
    eer, thr = compute_eer(y_true, y_score)
    try:
        auc = roc_auc_score(y_true, y_score)
    except ValueError:
        auc = float("nan")
    return y_true, y_score, eer, thr, auc


def domain_breakdown(y_true: np.ndarray, y_score: np.ndarray,
                      domains: np.ndarray, threshold: float) -> Dict[str, dict]:
    """
    Per-domain diagnostics at a fixed (global) operating threshold.
    Domains with both classes get their own EER/AUC.
    Single-class domains (e.g. librispeech real-only) get misclassification
    rate at the threshold - this surfaces environment-shortcut bias.
    """
    report: Dict[str, dict] = {}
    preds = (y_score >= threshold).astype(np.float32)
    for domain in sorted(set(domains.tolist())):
        mask = domains == domain
        yt, ys, pr = y_true[mask], y_score[mask], preds[mask]
        n = int(mask.sum())
        entry: Dict[str, Optional[float]] = {"n": n, "eer": None, "auc": None}
        if n > 0:
            entry["error_rate_at_thr"] = float(np.mean(pr != yt))
        else:
            entry["error_rate_at_thr"] = None
        if len(set(yt.tolist())) >= 2:
            try:
                d_eer, _ = compute_eer(yt, ys)
                entry["eer"] = d_eer
            except Exception:
                pass
            try:
                entry["auc"] = float(roc_auc_score(yt, ys))
            except ValueError:
                pass
        report[domain] = entry
    return report


def print_domain_breakdown(title: str, report: Dict[str, dict]) -> None:
    print(f"  {title}")
    for domain, m in report.items():
        if m["eer"] is not None:
            print(f"    {domain:<14} n={m['n']:5d}  EER={m['eer']*100:5.2f}%  "
                  f"AUC={m['auc']:.4f}  err@thr={m['error_rate_at_thr']*100:5.2f}%")
        else:
            print(f"    {domain:<14} n={m['n']:5d}  (single class)  "
                  f"err@thr={m['error_rate_at_thr']*100:5.2f}%")


# ===========================================================
# Audio transforms
# ===========================================================
_TELEPHONE_SOS = None

def _get_telephone_sos(sr: int = 16000):
    global _TELEPHONE_SOS
    if _TELEPHONE_SOS is None and HAS_SCIPY:
        _TELEPHONE_SOS = butter(4, [300, 3400], btype="band", fs=sr, output="sos")
    return _TELEPHONE_SOS


def telephone_bandlimit(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """Approximate PSTN/VoIP band-limit (300-3400 Hz)."""
    if not HAS_SCIPY or len(waveform) < 64:
        return waveform
    sos = _get_telephone_sos(sr)
    filtered = sosfilt(sos, waveform).astype(np.float32)
    peak = np.max(np.abs(filtered))
    if peak > 1e-8:
        filtered = filtered / peak
    return filtered


def mulaw_compress(waveform: np.ndarray, mu: int = 255) -> np.ndarray:
    """Light mu-law style non-linearity (G.711-ish)."""
    waveform = np.clip(waveform, -1.0, 1.0)
    compressed = np.sign(waveform) * np.log1p(mu * np.abs(waveform)) / np.log1p(mu)
    return compressed.astype(np.float32)


def deterministic_telephone_transform(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """No randomness - used to build the 'as-deployed' validation/holdout view."""
    w = telephone_bandlimit(waveform, sr=sr)
    w = mulaw_compress(w)
    return np.clip(w, -1.0, 1.0).astype(np.float32)


def simple_augment(waveform: np.ndarray, sr: int = 16000) -> np.ndarray:
    """
    Random telephony-oriented augmentation applied identically to every
    domain during TRAINING ONLY. Keeping this identical across domains is
    what stops the model from using "which corpus does this sound like"
    as a shortcut.
    """
    if random.random() < 0.5:
        gain = random.uniform(0.7, 1.3)
        waveform = waveform * gain
    if random.random() < 0.45:
        noise_level = random.uniform(0.001, 0.018)
        waveform = waveform + np.random.randn(len(waveform)).astype(np.float32) * noise_level
    if random.random() < 0.65:
        waveform = telephone_bandlimit(waveform, sr=sr)
    if random.random() < 0.40:
        waveform = mulaw_compress(waveform)
    if random.random() < 0.25:
        k = random.choice([3, 5, 7])
        kernel = np.ones(k, dtype=np.float32) / k
        if random.random() < 0.5:
            waveform = np.convolve(waveform, kernel, mode="same").astype(np.float32)
        else:
            low = np.convolve(waveform, kernel, mode="same")
            waveform = (waveform - low).astype(np.float32)
    return np.clip(waveform, -1.0, 1.0)


# ===========================================================
# Model (lightweight, for live-call / CPU inference)
# ===========================================================
class LiveCallSpoofDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=80, stride=4, padding=38),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(4),
            nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x)).squeeze(1)


# ===========================================================
# Domain-aware data discovery
# ===========================================================
def discover_domain_files(root: Path, domains: List[str]) -> Dict[str, List[str]]:
    domain_files = {}
    for domain in domains:
        folder = root / domain
        if not folder.is_dir():
            print(f"  {root.name}/{domain:<15} -> MISSING (folder not found: {folder})")
            domain_files[domain] = []
            continue
        files = sorted(glob.glob(str(folder / "*.npy")))
        if not files:
            print(f"  {root.name}/{domain:<15} -> 0 files (folder exists but is empty)")
        else:
            print(f"  {root.name}/{domain:<15} -> {len(files):>6} files")
        domain_files[domain] = files
    return domain_files


def split_domain_files(
    files: List[str],
    val_ratio: float,
    seed: int,
    min_files_for_split: int,
    group_regex: Optional[str],
    domain_name: str,
) -> Tuple[List[str], List[str]]:
    """Per-domain train/val split, optionally speaker/session-grouped."""
    if len(files) < min_files_for_split:
        return files, []
    if group_regex:
        pattern = re.compile(group_regex)
        groups = []
        fallback_count = 0
        for f in files:
            stem = Path(f).stem
            m = pattern.match(stem)
            if m and m.groups():
                groups.append(m.group(1))
            else:
                groups.append(stem)
                fallback_count += 1
        if fallback_count > 0:
            print(f"    [warn] {domain_name}: {fallback_count}/{len(files)} filenames did not "
                  f"match --speaker_group_regex; treated as their own group.")
        if len(set(groups)) >= 2:
            gss = GroupShuffleSplit(n_splits=1, test_size=val_ratio, random_state=seed)
            idx_train, idx_val = next(gss.split(files, groups=groups))
            return [files[i] for i in idx_train], [files[i] for i in idx_val]
        print(f"    [warn] {domain_name}: grouping produced <2 unique groups; "
              f"falling back to a plain file-level split.")
    return train_test_split(files, test_size=val_ratio, random_state=seed)


# ===========================================================
# Weighted, exact-size domain allocation
# ===========================================================
def parse_weight_string(s: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"Malformed weight spec '{part}' (expected domain:weight)")
        domain, w = part.split(":", 1)
        domain = domain.strip()
        try:
            weights[domain] = float(w.strip())
        except ValueError:
            raise ValueError(f"Malformed weight value in '{part}'")
    return weights


def parse_domain_list(s: str) -> List[str]:
    return [d.strip() for d in s.split(",") if d.strip()]


def filter_weights(weights: Dict[str, float], domain_sizes: Dict[str, int]) -> Dict[str, float]:
    """Keep only domains that actually have files and a positive weight."""
    return {d: w for d, w in weights.items() if domain_sizes.get(d, 0) > 0 and w > 0}


def allocate_by_weight(weights: Dict[str, float], target_total: int) -> Dict[str, int]:
    """Exact largest-remainder allocation: sum(result.values()) == target_total."""
    if target_total <= 0 or not weights:
        return {d: 0 for d in weights}
    total_w = sum(weights.values())
    if total_w <= 0:
        base = target_total // len(weights)
        alloc = {d: base for d in weights}
        rem = target_total - base * len(weights)
        for d in list(weights.keys())[:rem]:
            alloc[d] += 1
        return alloc
    raw = {d: target_total * (w / total_w) for d, w in weights.items()}
    floor_alloc = {d: int(np.floor(v)) for d, v in raw.items()}
    remainder = target_total - sum(floor_alloc.values())
    fracs = sorted(raw.keys(), key=lambda d: raw[d] - floor_alloc[d], reverse=True)
    for d in fracs[:remainder]:
        floor_alloc[d] += 1
    return floor_alloc


def allocate_with_caps(weights: Dict[str, float], caps: Dict[str, int],
                        target_total: int) -> Dict[str, int]:
    """Proportional allocation respecting per-domain caps (used for validation)."""
    active = {d: w for d, w in weights.items() if caps.get(d, 0) > 0}
    if not active:
        return {}
    remaining_total = min(target_total, sum(caps.values()))
    alloc = {d: 0 for d in active}
    free = set(active.keys())
    remaining = remaining_total
    while remaining > 0 and free:
        w_sum = sum(active[d] for d in free)
        if w_sum <= 0:
            share = {d: remaining // len(free) for d in free}
        else:
            share = {d: int(round(remaining * (active[d] / w_sum))) for d in free}
        progressed = False
        for d in list(free):
            room = caps[d] - alloc[d]
            give = min(share.get(d, 0), room)
            if give > 0:
                alloc[d] += give
                remaining -= give
                progressed = True
            if alloc[d] >= caps[d]:
                free.discard(d)
        if not progressed:
            break
    return alloc


class DomainCyclicSampler:
    """
    Per-domain, without-replacement-within-a-pass sampler.
    Every file is drawn exactly once before any file repeats.
    Guarantees exact block sizes and verifiable coverage.
    """
    def __init__(self, domain_files: Dict[str, List[str]], seed: int):
        self._rng = random.Random(seed)
        self._pools: Dict[str, List[str]] = {}
        self._cursors: Dict[str, int] = {}
        self._passes: Dict[str, int] = {}
        for domain, files in domain_files.items():
            pool = list(files)
            self._rng.shuffle(pool)
            self._pools[domain] = pool
            self._cursors[domain] = 0
            self._passes[domain] = 0

    def draw(self, domain: str, n: int) -> List[str]:
        pool = self._pools.get(domain, [])
        if not pool or n <= 0:
            return []
        out: List[str] = []
        cursor = self._cursors[domain]
        while len(out) < n:
            remaining = len(pool) - cursor
            take = min(remaining, n - len(out))
            out.extend(pool[cursor: cursor + take])
            cursor += take
            if cursor >= len(pool):
                self._rng.shuffle(pool)
                cursor = 0
                self._passes[domain] += 1
        self._cursors[domain] = cursor
        return out

    def passes_completed(self, domain: str) -> int:
        return self._passes.get(domain, 0)


def sample_val_domains(
    domain_files: Dict[str, List[str]],
    weights: Dict[str, float],
    target_total: int,
    seed: int,
) -> List[Tuple[str, str]]:
    """Fixed domain-weighted sample for building the validation set once."""
    domain_sizes = {d: len(v) for d, v in domain_files.items()}
    active_weights = filter_weights(weights, domain_sizes)
    if not active_weights:
        return []
    counts = allocate_with_caps(active_weights, domain_sizes, target_total)
    rng = random.Random(seed)
    selected: List[Tuple[str, str]] = []
    for domain, n in counts.items():
        pool = domain_files[domain]
        chosen = rng.sample(pool, min(n, len(pool)))
        selected.extend((f, domain) for f in chosen)
    rng.shuffle(selected)
    return selected


# ===========================================================
# Waveform loading
# ===========================================================
def load_waveforms(
    items: List[Tuple[str, str]],
    label: int,
    max_samples: int,
    augment: bool = False,
    crop_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """items: list of (filepath, domain_name) tuples."""
    data, labels, domains = [], [], []
    rng = random.Random(crop_seed) if crop_seed is not None else random
    for f, domain in items:
        try:
            w = np.load(f).astype(np.float32)
            if w.ndim != 1 or w.size == 0:
                print(f"Skipping {f}: empty or non-1D array (shape={w.shape})")
                continue
            if not np.isfinite(w).all():
                print(f"Skipping {f}: contains NaN/Inf values")
                continue
            peak = np.max(np.abs(w))
            if peak > 1e-8:
                w = w / peak
            if len(w) > max_samples:
                start = rng.randint(0, len(w) - max_samples)
                w = w[start: start + max_samples]
            elif len(w) < max_samples:
                w = np.pad(w, (0, max_samples - len(w)), mode="constant")
            if augment:
                w = simple_augment(w)
            data.append(w)
            labels.append(label)
            domains.append(domain)
        except Exception as e:
            print(f"Skipping {f}: {e}")
    if not data:
        return (np.zeros((0, max_samples), dtype=np.float32),
                np.zeros(0, dtype=np.float32),
                np.array([], dtype=object))
    return (np.asarray(data, dtype=np.float32),
            np.asarray(labels, dtype=np.float32),
            np.asarray(domains, dtype=object))


# ===========================================================
# Argument parsing / validation
# ===========================================================
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Multi-domain live-call spoof detector (hardened, exact counts)"
    )
    parser.add_argument("--real_dir", type=str,
                        default=r"G:\My Drive\Audio_Training_Data\real",
                        help="Root folder containing asvspoof/, in_the_wild/, librispeech/")
    parser.add_argument("--spoof_dir", type=str,
                        default=r"G:\My Drive\Audio_Training_Data\spoof",
                        help="Root folder containing asvspoof/, in_the_wild/")
    parser.add_argument("--cache_dir", type=str,
                        default=r"C:\sih-voice-detector\local_cache")

    # Domain lists match the actual inspection
    parser.add_argument("--real_domains", type=str,
                        default="asvspoof,in_the_wild,librispeech")
    parser.add_argument("--spoof_domains", type=str,
                        default="asvspoof,in_the_wild")

    # Weights tuned for the real size imbalance (74k librispeech vs ~20k in_the_wild)
    # Librispeech is deliberately capped so it cannot dominate.
    parser.add_argument("--real_domain_weights", type=str,
                        default="in_the_wild:0.50,librispeech:0.25,asvspoof:0.25",
                        help="domain:weight,...  (librispeech capped at 0.25)")
    parser.add_argument("--spoof_domain_weights", type=str,
                        default="in_the_wild:0.45,asvspoof:0.55")

    parser.add_argument("--max_samples", type=int, default=16000,
                        help="Window length (16000 = 1 s @ 16 kHz)")
    parser.add_argument("--block_real", type=int, default=1200)
    parser.add_argument("--block_spoof", type=int, default=1200)
    parser.add_argument("--blocks_per_epoch", type=int, default=12)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_ratio", type=float, default=0.10)
    parser.add_argument("--val_target_real", type=int, default=1200)
    parser.add_argument("--val_target_spoof", type=int, default=1200)
    parser.add_argument("--min_domain_files_for_split", type=int, default=10)
    parser.add_argument("--speaker_group_regex", type=str, default=None,
                        help="Optional regex with one capture group for speaker/session id")
    parser.add_argument("--select_metric", type=str, default="telephone",
                        choices=["clean", "telephone", "mean"],
                        help="Which validation EER drives checkpointing/early stopping")
    parser.add_argument("--external_real_dir", type=str, default=None)
    parser.add_argument("--external_spoof_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_cuda", action="store_true")
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    def require(cond: bool, msg: str) -> None:
        if not cond:
            parser.error(msg)
    require(0.0 < args.val_ratio < 1.0, "--val_ratio must be in (0, 1)")
    require(args.max_samples > 0, "--max_samples must be > 0")
    require(args.block_real > 0, "--block_real must be > 0")
    require(args.block_spoof > 0, "--block_spoof must be > 0")
    require(args.blocks_per_epoch > 0, "--blocks_per_epoch must be > 0")
    require(args.num_epochs > 0, "--num_epochs must be > 0")
    require(args.patience > 0, "--patience must be > 0")
    require(args.batch_size > 0, "--batch_size must be > 0")
    require(args.lr > 0, "--lr must be > 0")
    require(args.val_target_real > 0, "--val_target_real must be > 0")
    require(args.val_target_spoof > 0, "--val_target_spoof must be > 0")
    require(args.min_domain_files_for_split >= 2, "--min_domain_files_for_split must be >= 2")
    require(bool(args.real_dir) and bool(args.spoof_dir), "--real_dir/--spoof_dir must be set")
    require((args.external_real_dir is None) == (args.external_spoof_dir is None),
            "--external_real_dir and --external_spoof_dir must be provided together")
    if args.speaker_group_regex:
        try:
            re.compile(args.speaker_group_regex)
        except re.error as e:
            parser.error(f"--speaker_group_regex is not a valid regex: {e}")
    for flag in ("real_domain_weights", "spoof_domain_weights"):
        try:
            parse_weight_string(getattr(args, flag))
        except ValueError as e:
            parser.error(f"--{flag}: {e}")


# ===========================================================
# Main
# ===========================================================
def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)

    if args.use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using device: CUDA ({torch.cuda.get_device_name(0)})")
    else:
        if args.use_cuda and not torch.cuda.is_available():
            print("Warning: --use_cuda was set but no CUDA device is available; falling back to CPU.")
        device = torch.device("cpu")
        num_cores = os.cpu_count() or 4
        torch.set_num_threads(num_cores)
        torch.set_num_interop_threads(num_cores)
        print(f"Using device: CPU ({num_cores} cores)")

    real_root = Path(os.environ.get("REAL_DIR", args.real_dir))
    spoof_root = Path(os.environ.get("SPOOF_DIR", args.spoof_dir))
    cache_dir = Path(os.environ.get("CACHE_DIR", args.cache_dir))
    cache_dir.mkdir(parents=True, exist_ok=True)

    real_domains = parse_domain_list(args.real_domains)
    spoof_domains = parse_domain_list(args.spoof_domains)
    real_weights_cfg = parse_weight_string(args.real_domain_weights)
    spoof_weights_cfg = parse_weight_string(args.spoof_domain_weights)

    unknown_real_w = set(real_weights_cfg) - set(real_domains)
    unknown_spoof_w = set(spoof_weights_cfg) - set(spoof_domains)
    if unknown_real_w:
        print(f"Warning: --real_domain_weights mentions unknown domains {unknown_real_w}; ignored.")
    if unknown_spoof_w:
        print(f"Warning: --spoof_domain_weights mentions unknown domains {unknown_spoof_w}; ignored.")

    print("\n📂 Discovering domain folders (expected counts from inspection)...")
    print("   Real  expected: asvspoof=2,580 | in_the_wild=19,963 | librispeech=74,344")
    print("   Spoof expected: asvspoof=22,800 | in_the_wild=11,816 | librispeech=0")
    real_files = discover_domain_files(real_root, real_domains)
    spoof_files = discover_domain_files(spoof_root, spoof_domains)

    total_real = sum(len(v) for v in real_files.values())
    total_spoof = sum(len(v) for v in spoof_files.values())
    if total_real == 0 or total_spoof == 0:
        raise FileNotFoundError("No .npy files found in the expected domain sub-folders.")
    print(f"\nTotal discovered → Real: {total_real} | Spoof: {total_spoof}")

    # ---------------- Domain-aware train/val split ----------------
    if not args.speaker_group_regex:
        print("\n[note] --speaker_group_regex not set: split is file-level. "
              "If files share speakers/sessions this can leak and inflate EER.")

    real_train, real_val = {}, {}
    for domain, files in real_files.items():
        tr, va = split_domain_files(
            files, args.val_ratio, args.seed,
            args.min_domain_files_for_split,
            args.speaker_group_regex, f"real/{domain}"
        )
        real_train[domain], real_val[domain] = tr, va

    spoof_train, spoof_val = {}, {}
    for domain, files in spoof_files.items():
        tr, va = split_domain_files(
            files, args.val_ratio, args.seed,
            args.min_domain_files_for_split,
            args.speaker_group_regex, f"spoof/{domain}"
        )
        spoof_train[domain], spoof_val[domain] = tr, va

    print("\nTrain / Val counts per domain:")
    for d in real_domains:
        print(f"  Real/{d:<12} train={len(real_train.get(d, [])):5d}  val={len(real_val.get(d, [])):5d}")
    for d in spoof_domains:
        print(f"  Spoof/{d:<12} train={len(spoof_train.get(d, [])):5d}  val={len(spoof_val.get(d, [])):5d}")

    active_real_weights = filter_weights(
        real_weights_cfg, {d: len(real_train.get(d, [])) for d in real_domains}
    )
    active_spoof_weights = filter_weights(
        spoof_weights_cfg, {d: len(spoof_train.get(d, [])) for d in spoof_domains}
    )
    if not active_real_weights or not active_spoof_weights:
        raise RuntimeError(
            "No usable (weighted, non-empty) training domains for real or spoof. "
            "Check --real_domain_weights/--spoof_domain_weights against discovered folders."
        )
    print(f"\nActive real training weights:  {active_real_weights}")
    print(f"Active spoof training weights: {active_spoof_weights}")

    max_samples = args.max_samples

    # ---------------- Fixed validation set (clean + telephone) ----------------
    print("\nBuilding domain-balanced validation set...")
    val_real_items = sample_val_domains(
        real_val, real_weights_cfg, args.val_target_real, args.seed + 100
    )
    val_spoof_items = sample_val_domains(
        spoof_val, spoof_weights_cfg, args.val_target_spoof, args.seed + 200
    )
    val_real_data, val_real_y, val_real_dom = load_waveforms(
        val_real_items, 0, max_samples, augment=False, crop_seed=args.seed + 1000
    )
    val_spoof_data, val_spoof_y, val_spoof_dom = load_waveforms(
        val_spoof_items, 1, max_samples, augment=False, crop_seed=args.seed + 2000
    )
    X_val = np.concatenate([val_real_data, val_spoof_data]) if len(val_real_data) or len(val_spoof_data) \
        else np.zeros((0, max_samples), dtype=np.float32)
    y_val = np.concatenate([val_real_y, val_spoof_y])
    dom_val = np.concatenate([val_real_dom, val_spoof_dom])
    if X_val.shape[0] == 0 or len(set(y_val.tolist())) < 2:
        raise RuntimeError(
            "Validation set is empty or single-class after loading. "
            "Check val_ratio, val_target_real/spoof, and domain folder contents."
        )
    print(f"Validation size → Real: {len(val_real_items)} | Spoof: {len(val_spoof_items)}")

    X_val_t = torch.from_numpy(X_val).unsqueeze(1)
    y_val_t = torch.from_numpy(y_val)
    val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=args.batch_size, shuffle=False)

    X_val_tel = np.stack([deterministic_telephone_transform(w) for w in X_val]).astype(np.float32)
    val_loader_tel = DataLoader(
        TensorDataset(torch.from_numpy(X_val_tel).unsqueeze(1), y_val_t),
        batch_size=args.batch_size, shuffle=False
    )

    # ---------------- Model & optimiser ----------------
    model = LiveCallSpoofDetector().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)

    best_path = cache_dir / "best_live_call_spoof_model.pth"
    best_score = 1.0
    patience_counter = 0

    real_sampler = DomainCyclicSampler(real_train, seed=args.seed)
    spoof_sampler = DomainCyclicSampler(spoof_train, seed=args.seed + 1)

    # ---------------- Run manifest ----------------
    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "args": vars(args),
        "resolved_real_domain_weights": active_real_weights,
        "resolved_spoof_domain_weights": active_spoof_weights,
        "dataset_composition": {
            "real_train": {d: len(v) for d, v in real_train.items()},
            "real_val": {d: len(v) for d, v in real_val.items()},
            "spoof_train": {d: len(v) for d, v in spoof_train.items()},
            "spoof_val": {d: len(v) for d, v in spoof_val.items()},
            "inspection_totals": {
                "real_asvspoof": 2580, "real_in_the_wild": 19963, "real_librispeech": 74344,
                "spoof_asvspoof": 22800, "spoof_in_the_wild": 11816, "spoof_librispeech": 0,
            },
        },
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "scipy_available": HAS_SCIPY,
    }
    with open(cache_dir / "run_manifest.json", "w") as fh:
        json.dump(manifest, fh, indent=2, default=str)

    print(f"\nStarting training – up to {args.num_epochs} epochs "
          f"(early-stop patience={args.patience}, select_metric={args.select_metric})")
    print("Sampling policy: exact-size domain-balanced blocks via cyclic per-domain sampling")
    print("Librispeech Real weight capped at 0.25 to prevent domination by the 74k clean files\n")

    import time
    training_start = time.time()

    for epoch in range(args.num_epochs):
        epoch_start = time.time()
        overall_pct = 100.0 * epoch / max(args.num_epochs, 1)
        bar_len = 30
        filled = int(bar_len * epoch / max(args.num_epochs, 1))
        bar = "█" * filled + "░" * (bar_len - filled)

        print("=" * 60)
        print(f"Epoch {epoch + 1}/{args.num_epochs}  [{bar}] {overall_pct:5.1f}%")
        print("=" * 60)

        model.train()
        epoch_correct = epoch_total = 0
        nan_batches = 0
        n_blocks = args.blocks_per_epoch

        for b_idx in range(n_blocks):
            block_start = time.time()
            real_counts = allocate_by_weight(active_real_weights, args.block_real)
            spoof_counts = allocate_by_weight(active_spoof_weights, args.block_spoof)

            curr_real_items = [(f, d) for d, n in real_counts.items() for f in real_sampler.draw(d, n)]
            curr_spoof_items = [(f, d) for d, n in spoof_counts.items() for f in spoof_sampler.draw(d, n)]

            # Progress while loading (can be slow)
            if not HAS_TQDM:
                print(f"  ▸ Block {b_idx + 1}/{n_blocks}: loading waveforms...", end="", flush=True)

            real_data, real_y, _ = load_waveforms(curr_real_items, 0, max_samples, augment=True)
            spoof_data, spoof_y, _ = load_waveforms(curr_spoof_items, 1, max_samples, augment=True)

            if len(real_data) == 0 or len(spoof_data) == 0:
                print(f"\r  ▸ Block {b_idx + 1}/{n_blocks}: skipped (no usable samples on one side)")
                continue

            # Re-balance to exact 1:1 after possible load failures
            min_len = min(len(real_data), len(spoof_data))
            if min_len < len(real_data) or min_len < len(spoof_data):
                idx_r = np.random.choice(len(real_data), min_len, replace=False)
                idx_s = np.random.choice(len(spoof_data), min_len, replace=False)
                real_data, real_y = real_data[idx_r], real_y[idx_r]
                spoof_data, spoof_y = spoof_data[idx_s], spoof_y[idx_s]

            X = torch.from_numpy(np.concatenate([real_data, spoof_data])).unsqueeze(1)
            y = torch.from_numpy(np.concatenate([real_y, spoof_y]))

            train_loader = DataLoader(
                TensorDataset(X, y),
                batch_size=args.batch_size,
                shuffle=True,
                drop_last=True,
                num_workers=args.num_workers,
            )

            n_batches = len(train_loader)
            loop = tqdm(
                train_loader,
                desc=f"  Epoch {epoch+1}/{args.num_epochs} │ Block {b_idx+1}/{n_blocks}",
                total=n_batches,
                leave=False,
                disable=not HAS_TQDM,
                unit="batch",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}",
            )
            for waveforms, labels in loop:
                waveforms, labels = waveforms.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(waveforms)
                loss = criterion(logits, labels)
                if not torch.isfinite(loss):
                    nan_batches += 1
                    optimizer.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                preds = (torch.sigmoid(logits) >= 0.5).float()
                epoch_correct += (preds == labels).sum().item()
                epoch_total += labels.size(0)
                if HAS_TQDM:
                    loop.set_postfix(
                        loss=f"{loss.item():.4f}",
                        acc=f"{100 * epoch_correct / max(epoch_total, 1):.1f}%",
                    )

            block_elapsed = time.time() - block_start
            acc = 100.0 * epoch_correct / max(epoch_total, 1)
            cov_r = {d: real_sampler.passes_completed(d) for d in active_real_weights}
            cov_s = {d: spoof_sampler.passes_completed(d) for d in active_spoof_weights}

            # Always print a clear one-line block summary
            print(f"  ✓ Block {b_idx+1}/{n_blocks} done  "
                  f"acc={acc:.1f}%  "
                  f"time={block_elapsed:.1f}s  "
                  f"passes(real)={cov_r}  passes(spoof)={cov_s}")

        if nan_batches > 0:
            print(f"  [warn] {nan_batches} batch(es) had non-finite loss this epoch and were skipped.")

        scheduler.step()

        # ---------------- Validation ----------------
        print("  ▸ Running validation (clean + telephone)...")
        yt_c, ys_c, eer_c, thr_c, auc_c = evaluate(model, val_loader, device)
        yt_t, ys_t, eer_t, thr_t, auc_t = evaluate(model, val_loader_tel, device)

        print(f"Val[clean]     EER: {eer_c*100:.2f}% | AUC: {auc_c:.4f} | thr≈{thr_c:.3f}")
        print(f"Val[telephone] EER: {eer_t*100:.2f}% | AUC: {auc_t:.4f} | thr≈{thr_t:.3f}")
        print_domain_breakdown("Per-domain (clean):", domain_breakdown(yt_c, ys_c, dom_val, thr_c))
        print_domain_breakdown("Per-domain (telephone):", domain_breakdown(yt_t, ys_t, dom_val, thr_t))

        # Epoch timing + ETA
        epoch_elapsed = time.time() - epoch_start
        total_elapsed = time.time() - training_start
        epochs_done = epoch + 1
        avg_epoch = total_elapsed / epochs_done
        remaining_epochs = args.num_epochs - epochs_done
        eta_sec = avg_epoch * remaining_epochs
        eta_min = eta_sec / 60.0
        print(f"  ⏱ Epoch time: {epoch_elapsed:.1f}s  |  "
              f"Total: {total_elapsed/60:.1f} min  |  "
              f"ETA: ~{eta_min:.1f} min ({remaining_epochs} epochs left)")

        if args.select_metric == "clean":
            current_score = eer_c
        elif args.select_metric == "telephone":
            current_score = eer_t
        else:
            current_score = (eer_c + eer_t) / 2.0

        if current_score < best_score - 1e-4:
            best_score = current_score
            patience_counter = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "select_metric": args.select_metric,
                "score": best_score,
                "clean_eer": eer_c, "clean_auc": auc_c, "clean_threshold": thr_c,
                "telephone_eer": eer_t, "telephone_auc": auc_t, "telephone_threshold": thr_t,
                "epoch": epoch,
                "args": vars(args),
                "real_weights": active_real_weights,
                "spoof_weights": active_spoof_weights,
            }, best_path)
            print(f"  → New best model saved ({args.select_metric} EER {best_score*100:.2f}%)")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{args.patience})")
            if patience_counter >= args.patience:
                print("Early stopping triggered.")
                break

    print(f"\nTraining finished. Best {args.select_metric} EER: {best_score*100:.2f}%")
    print(f"Best checkpoint: {best_path}")

    # ---------------- One-time external holdout evaluation ----------------
    if args.external_real_dir and args.external_spoof_dir:
        print("\n" + "=" * 60)
        print("FINAL EXTERNAL HOLDOUT EVALUATION "
              "(never used for training or model selection)")
        print("=" * 60)
        ext_real_files = sorted(glob.glob(str(Path(args.external_real_dir) / "*.npy")))
        ext_spoof_files = sorted(glob.glob(str(Path(args.external_spoof_dir) / "*.npy")))
        if not ext_real_files or not ext_spoof_files:
            print(f"  [warn] Skipped: no .npy files found in "
                  f"{args.external_real_dir} or {args.external_spoof_dir}")
        else:
            if best_path.exists():
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"  Loaded best checkpoint (epoch {ckpt.get('epoch')})")
            else:
                print("  [warn] No best checkpoint found; evaluating current in-memory weights.")
            ext_real_items = [(f, "external") for f in ext_real_files]
            ext_spoof_items = [(f, "external") for f in ext_spoof_files]
            ext_real_data, ext_real_y, _ = load_waveforms(
                ext_real_items, 0, max_samples, augment=False, crop_seed=args.seed + 9000
            )
            ext_spoof_data, ext_spoof_y, _ = load_waveforms(
                ext_spoof_items, 1, max_samples, augment=False, crop_seed=args.seed + 9001
            )
            if len(ext_real_data) == 0 or len(ext_spoof_data) == 0:
                print("  [warn] External holdout produced zero usable samples on one side.")
            else:
                X_ext = np.concatenate([ext_real_data, ext_spoof_data])
                y_ext = np.concatenate([ext_real_y, ext_spoof_y])
                X_ext_t = torch.from_numpy(X_ext).unsqueeze(1)
                y_ext_t = torch.from_numpy(y_ext)
                ext_loader = DataLoader(TensorDataset(X_ext_t, y_ext_t),
                                        batch_size=args.batch_size, shuffle=False)
                X_ext_tel = np.stack([deterministic_telephone_transform(w) for w in X_ext]).astype(np.float32)
                ext_loader_tel = DataLoader(
                    TensorDataset(torch.from_numpy(X_ext_tel).unsqueeze(1), y_ext_t),
                    batch_size=args.batch_size, shuffle=False
                )
                _, _, ext_eer_c, _, ext_auc_c = evaluate(model, ext_loader, device)
                _, _, ext_eer_t, _, ext_auc_t = evaluate(model, ext_loader_tel, device)
                print(f"  External[clean]     EER: {ext_eer_c*100:.2f}% | AUC: {ext_auc_c:.4f}")
                print(f"  External[telephone] EER: {ext_eer_t*100:.2f}% | AUC: {ext_auc_t:.4f}")
                report_path = cache_dir / "external_holdout_report.json"
                with open(report_path, "w") as fh:
                    json.dump({
                        "n_real": len(ext_real_files), "n_spoof": len(ext_spoof_files),
                        "clean_eer": ext_eer_c, "clean_auc": ext_auc_c,
                        "telephone_eer": ext_eer_t, "telephone_auc": ext_auc_t,
                    }, fh, indent=2)
                print(f"  Report saved to {report_path}")

    print(
        "\nAnti-bias / robustness measures applied:\n"
        "  • Exact-size, domain-balanced training blocks (cyclic sampler)\n"
        "  • Librispeech Real weight capped at 0.25 (prevents 74k domination)\n"
        "  • Domain-stratified validation (clean + telephone-simulated)\n"
        "  • Per-domain EER / error-rate breakdown every epoch\n"
        "  • Post-load 1:1 rebalancing after possible file skips\n"
        "  • Run manifest + best/external-holdout reports written to cache_dir\n"
        "Remember: final evaluation should still use a completely external "
        "telephony / modern-deepfake test set."
    )


if __name__ == "__main__":
    arg_parser = build_arg_parser()
    parsed_args = arg_parser.parse_args()
    validate_args(arg_parser, parsed_args)
    try:
        main(parsed_args)
    except KeyboardInterrupt:
        print("\n[Interrupted by user] Exiting. If a best checkpoint had already been "
              "saved, it is safe to use as-is.")
        sys.exit(130)
