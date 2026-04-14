"""
renderer.py — Publication-quality spectrogram export and rolling metrics CSV.

Outputs
-------
  spectrogram_<timestamp>.png   — high-DPI dual-panel figure (raw stream + features)
  metrics_<timestamp>.csv       — one row per FeatureFrame with all feature values
"""

from __future__ import annotations
import csv
import time
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for file export
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import Normalize
from typing import List
from features import FeatureFrame

# Publication style parameters
DPI = 300
FIGURE_SIZE = (14, 8)
COLORMAP = "inferno"
FONT_FAMILY = "DejaVu Sans"

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": DPI,
})


class Renderer:
    """
    Collects FeatureFrames and raw samples, then exports:
    - Rolling metrics to CSV (append mode, flushed per frame)
    - Spectrogram PNG on demand or at interval
    """

    def __init__(self, output_dir: str = "."):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        self._csv_path = os.path.join(output_dir, f"metrics_{ts}.csv")
        self._png_base = os.path.join(output_dir, f"spectrogram_{ts}")

        self._frames: List[FeatureFrame] = []
        self._raw_samples: List[np.ndarray] = []
        self._csv_file = open(self._csv_path, "w", newline="")
        self._csv_writer = csv.writer(self._csv_file)
        self._csv_writer.writerow([
            "frame_index", "timestamp",
            "entropy", "bias", "autocorr_lag1",
            "runs_z", "hurst", "spectral_flat",
        ])
        print(f"[renderer] CSV -> {self._csv_path}")

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def push_frame(self, frame: FeatureFrame, raw: np.ndarray) -> None:
        """Ingest one FeatureFrame and append to CSV immediately."""
        self._frames.append(frame)
        self._raw_samples.append(raw.copy())

        idx = len(self._frames) - 1
        self._csv_writer.writerow([
            idx,
            f"{frame.timestamp:.1f}",
            f"{frame.entropy:.6f}",
            f"{frame.bias:.6f}",
            f"{frame.autocorr_lag1:.6f}",
            f"{frame.runs_z:.6f}",
            f"{frame.hurst:.6f}",
            f"{frame.spectral_flat:.6f}",
        ])
        self._csv_file.flush()

    def render_spectrogram(self, suffix: str = "") -> str:
        """
        Export a dual-panel spectrogram PNG and return its path.

        Panel 1: Raw sample stream as 2-D time-frequency spectrogram
        Panel 2: Feature timeline (all 6 metrics stacked)
        """
        if len(self._frames) < 4:
            print("[renderer] Not enough frames to render yet.")
            return ""

        filename = f"{self._png_base}{suffix}.png"
        frames = self._frames
        n = len(frames)

        # ── Assemble raw stream array ─────────────────────────────────
        raw_concat = np.concatenate([f.raw_window for f in frames]).astype(float)

        # ── Feature timeline arrays ───────────────────────────────────
        t = np.arange(n)
        entropy       = np.array([f.entropy       for f in frames])
        bias          = np.array([f.bias          for f in frames])
        autocorr      = np.array([f.autocorr_lag1 for f in frames])
        runs_z        = np.array([f.runs_z        for f in frames]) / 5.0  # normalise
        hurst         = np.array([f.hurst         for f in frames])
        spectral_flat = np.array([f.spectral_flat for f in frames])

        # ── Figure layout ─────────────────────────────────────────────
        fig = plt.figure(figsize=FIGURE_SIZE)
        gs = gridspec.GridSpec(
            3, 2,
            figure=fig,
            height_ratios=[2, 1, 1],
            hspace=0.45,
            wspace=0.3,
        )

        ax_spec  = fig.add_subplot(gs[0, :])   # full-width spectrogram
        ax_ent   = fig.add_subplot(gs[1, 0])   # entropy + spectral flat
        ax_bias  = fig.add_subplot(gs[1, 1])   # bias
        ax_ac    = fig.add_subplot(gs[2, 0])   # autocorrelation
        ax_hurst = fig.add_subplot(gs[2, 1])   # hurst + runs_z

        # ── Panel 1: Spectrogram ──────────────────────────────────────
        NFFT = 256
        hop = NFFT // 4
        ax_spec.specgram(
            raw_concat,
            NFFT=NFFT,
            Fs=1.0,          # sample-indexed, not time-indexed
            noverlap=NFFT - hop,
            cmap=COLORMAP,
            scale="dB",
            mode="magnitude",
        )
        ax_spec.set_title("QRNG Stream Spectrogram (ANU API - uint8)", fontweight="bold")
        ax_spec.set_xlabel("Sample index")
        ax_spec.set_ylabel("Frequency bin")
        ax_spec.set_ylim(0, 0.5)

        # ── Panel 2: Entropy + Spectral Flatness ──────────────────────
        ax_ent.plot(t, entropy, color="#e67e22", lw=1.5, label="Entropy")
        ax_ent.plot(t, spectral_flat, color="#3498db", lw=1.0, ls="--", label="Spectral flat.")
        ax_ent.axhline(1.0, color="gray", lw=0.5, ls=":")
        ax_ent.set_ylim(0, 1.1)
        ax_ent.set_ylabel("Value [0,1]")
        ax_ent.set_title("Entropy & Spectral Flatness")
        ax_ent.legend(fontsize=7, loc="lower right")

        # ── Panel 3: Bias ─────────────────────────────────────────────
        ax_bias.fill_between(t, 0, bias, where=bias >= 0, color="#e74c3c", alpha=0.6, label="+bias")
        ax_bias.fill_between(t, 0, bias, where=bias < 0,  color="#2ecc71", alpha=0.6, label="-bias")
        ax_bias.axhline(0, color="gray", lw=0.8)
        ax_bias.set_ylim(-1.1, 1.1)
        ax_bias.set_ylabel("Normalised bias")
        ax_bias.set_title("Mean Bias Deviation")
        ax_bias.legend(fontsize=7, loc="upper right")

        # ── Panel 4: Autocorrelation ──────────────────────────────────
        ax_ac.plot(t, autocorr, color="#9b59b6", lw=1.5)
        ax_ac.axhline(0, color="gray", lw=0.8)
        ax_ac.set_ylim(-1.1, 1.1)
        ax_ac.set_ylabel("Lag-1 autocorr.")
        ax_ac.set_xlabel("Frame index")
        ax_ac.set_title("Autocorrelation (lag 1)")

        # ── Panel 5: Hurst + Runs Z ───────────────────────────────────
        ax_hurst.plot(t, hurst, color="#1abc9c", lw=1.5, label="Hurst H")
        ax_hurst.plot(t, runs_z, color="#e74c3c", lw=1.0, ls=":", label="Runs z/5")
        ax_hurst.axhline(0.5, color="gray", lw=0.5, ls=":")
        ax_hurst.set_ylim(-1.1, 1.1)
        ax_hurst.set_ylabel("Value")
        ax_hurst.set_xlabel("Frame index")
        ax_hurst.set_title("Hurst Exponent & Runs Test Z-score")
        ax_hurst.legend(fontsize=7, loc="upper right")

        # ── Shared annotation ─────────────────────────────────────────
        fig.text(
            0.5, 0.01,
            f"Source: ANU QRNG API  |  Frames: {n}  |  "
            f"Window: {frames[0].raw_window.shape[0]} samples  |  "
            f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
            ha="center", va="bottom", fontsize=7, color="#555555",
        )

        plt.savefig(filename, dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"[renderer] Spectrogram -> {filename}")
        return filename

    def close(self) -> None:
        self._csv_file.close()
        print(f"[renderer] CSV closed: {self._csv_path}")
