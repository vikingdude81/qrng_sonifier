"""
anomaly_triggers.py — Threshold-based anomaly detection with audio events.

Monitors incoming FeatureFrames (from any source) for statistically
interesting conditions and fires:
  - a distinct synthesised audio event (chime, click, tone burst)
  - a structured log entry in anomaly_log.csv
  - an optional callback for downstream use (dashboard, alert, etc.)

Trigger catalogue
─────────────────
  entropy_drop       entropy falls below threshold (e.g. < 0.85)
  entropy_surge      entropy unexpectedly high (> 0.995)
  bias_drift         |bias| exceeds threshold for N consecutive frames
  autocorr_spike     |autocorr_lag1| exceeds threshold (persistent structure)
  runs_anomaly       |runs_z| > 3  (non-random run pattern)
  hurst_persist      hurst > 0.65  (persistent/correlated stream)
  hurst_antipersist  hurst < 0.35  (anti-persistent / mean-reverting)
  spectral_narrow    spectral_flat < 0.5  (histogram has dominant peaks)
  source_diverge     KL divergence between QRNG and PRNG windows exceeds threshold
  regime_change      composite score jump across consecutive windows
"""

from __future__ import annotations
import csv
import logging
import os
import time
import threading
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Dict, Any
from features import FeatureFrame

logger = logging.getLogger(__name__)

try:
    import sounddevice as sd
    AUDIO_OK = True
except ImportError:
    AUDIO_OK = False


SAMPLE_RATE = 44100

# ── Trigger definitions ───────────────────────────────────────────────────── #

@dataclass
class TriggerConfig:
    name: str
    enabled: bool = True
    # Threshold value - meaning depends on trigger type
    threshold: float = 0.0
    # How many consecutive frames must satisfy condition before firing
    consecutive: int = 1
    # Minimum seconds between re-fires of the same trigger (cool-down)
    cooldown: float = 5.0
    # Audio event type: "chime", "click", "burst", "sweep", "low_tone"
    audio_event: str = "chime"
    # Severity: "info", "warning", "critical"
    severity: str = "warning"


DEFAULT_TRIGGERS: List[TriggerConfig] = [
    TriggerConfig("entropy_drop",      threshold=0.85, consecutive=3, cooldown=10.0,
                  audio_event="low_tone",  severity="critical"),
    TriggerConfig("entropy_surge",     threshold=0.999, consecutive=2, cooldown=15.0,
                  audio_event="chime",     severity="info"),
    TriggerConfig("bias_drift",        threshold=0.15,  consecutive=5, cooldown=8.0,
                  audio_event="sweep",     severity="warning"),
    TriggerConfig("autocorr_spike",    threshold=0.12,  consecutive=3, cooldown=6.0,
                  audio_event="burst",     severity="warning"),
    TriggerConfig("runs_anomaly",      threshold=3.0,   consecutive=2, cooldown=5.0,
                  audio_event="click",     severity="warning"),
    TriggerConfig("hurst_persist",     threshold=0.65,  consecutive=4, cooldown=10.0,
                  audio_event="sweep",     severity="warning"),
    TriggerConfig("hurst_antipersist", threshold=0.35,  consecutive=4, cooldown=10.0,
                  audio_event="sweep",     severity="info"),
    TriggerConfig("spectral_narrow",   threshold=0.50,  consecutive=3, cooldown=8.0,
                  audio_event="burst",     severity="warning"),
    TriggerConfig("source_diverge",    threshold=0.05,  consecutive=2, cooldown=12.0,
                  audio_event="chime",     severity="critical"),
    TriggerConfig("regime_change",     threshold=0.30,  consecutive=1, cooldown=5.0,
                  audio_event="click",     severity="warning"),
]


# ── Fired event record ────────────────────────────────────────────────────── #

@dataclass
class AnomalyEvent:
    trigger_name: str
    severity: str
    source: str           # "qrng" | "prng" | "divergence"
    frame_index: int
    timestamp: float
    feature_value: float
    threshold: float
    message: str
    extra: Dict[str, Any] = field(default_factory=dict)


# ── Audio event synthesis ─────────────────────────────────────────────────── #

def _synth_event(event_type: str, severity: str) -> np.ndarray:
    """
    Synthesise a short stereo audio event and return float32 ndarray (N, 2).

    event_type  duration  character
    ──────────  ────────  ─────────────────────────────────────────
    chime       0.6 s     decaying sine - info / positive anomaly
    click       0.05 s    sharp transient - subtle flag
    burst       0.3 s     noise burst - unexpected structure
    sweep       0.5 s     frequency sweep - drift / regime change
    low_tone    0.8 s     deep tone - critical / entropy drop
    """
    sr = SAMPLE_RATE
    severity_gain = {"info": 0.25, "warning": 0.45, "critical": 0.70}.get(severity, 0.4)

    if event_type == "chime":
        dur = 0.6
        t = np.linspace(0, dur, int(sr * dur))
        env = np.exp(-5 * t)
        sig = env * np.sin(2 * np.pi * 880 * t)
        sig += 0.4 * env * np.sin(2 * np.pi * 1320 * t)

    elif event_type == "click":
        dur = 0.05
        t = np.linspace(0, dur, int(sr * dur))
        env = np.exp(-80 * t)
        sig = env * np.sin(2 * np.pi * 2200 * t)

    elif event_type == "burst":
        dur = 0.3
        n = int(sr * dur)
        rng = np.random.default_rng()
        noise = rng.standard_normal(n)
        env = np.exp(-8 * np.linspace(0, dur, n))
        # bandpass-ish: modulate noise with a carrier
        carrier = np.sin(2 * np.pi * 600 * np.linspace(0, dur, n))
        sig = noise * env * carrier

    elif event_type == "sweep":
        dur = 0.5
        t = np.linspace(0, dur, int(sr * dur))
        f_start, f_end = (200, 1200) if severity != "critical" else (1200, 200)
        phase = 2 * np.pi * np.cumsum(np.linspace(f_start, f_end, len(t)) / sr)
        env = np.sin(np.pi * t / dur)  # Hann envelope
        sig = env * np.sin(phase)

    elif event_type == "low_tone":
        dur = 0.8
        t = np.linspace(0, dur, int(sr * dur))
        env = np.exp(-3 * t)
        sig = env * np.sin(2 * np.pi * 110 * t)
        sig += 0.5 * env * np.sin(2 * np.pi * 165 * t)

    else:
        dur = 0.2
        t = np.linspace(0, dur, int(sr * dur))
        sig = np.sin(2 * np.pi * 440 * t) * np.exp(-10 * t)

    sig = (sig * severity_gain).astype(np.float32)
    stereo = np.column_stack([sig, sig])
    return stereo


def _play_event(event_type: str, severity: str) -> None:
    """Fire-and-forget audio playback on a daemon thread."""
    if not AUDIO_OK:
        return
    def _play():
        audio = _synth_event(event_type, severity)
        try:
            sd.play(audio, samplerate=SAMPLE_RATE, blocking=True)
        except Exception as e:
            logger.warning("Audio playback failed for event '%s': %s", event_type, e)
    threading.Thread(target=_play, daemon=True).start()


# ── Main detector class ───────────────────────────────────────────────────── #

class AnomalyDetector:
    """
    Monitors FeatureFrames from one or two sources and fires trigger events.

    Parameters
    ----------
    triggers : list[TriggerConfig] or None
        Override trigger list.  None = DEFAULT_TRIGGERS.
    output_dir : str
        Where to write anomaly_log.csv.
    audio : bool
        Enable audio events.
    callback : callable or None
        Optional fn(AnomalyEvent) called on every fired trigger.
    """

    def __init__(
        self,
        triggers: Optional[List[TriggerConfig]] = None,
        output_dir: str = "./output",
        audio: bool = True,
        callback: Optional[Callable[[AnomalyEvent], None]] = None,
    ):
        self._triggers: Dict[str, TriggerConfig] = {
            t.name: t for t in (triggers or DEFAULT_TRIGGERS)
        }
        self._audio = audio and AUDIO_OK
        self._callback = callback

        # Per-trigger state
        self._consecutive: Dict[str, int] = {k: 0 for k in self._triggers}
        self._last_fire: Dict[str, float] = {k: 0.0 for k in self._triggers}

        # Frame history for regime change detection
        self._score_history: List[float] = []
        self._frame_index = 0
        self._prng_windows: List[np.ndarray] = []

        # CSV log
        os.makedirs(output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        self._log_path = os.path.join(output_dir, f"anomaly_log_{ts}.csv")
        self._csv = open(self._log_path, "w", newline="")
        self._writer = csv.writer(self._csv)
        self._writer.writerow([
            "timestamp", "frame_index", "trigger", "severity",
            "source", "feature_value", "threshold", "message",
        ])
        logger.info("Log -> %s", self._log_path)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def push(
        self,
        qrng_frame: FeatureFrame,
        prng_frame: Optional[FeatureFrame] = None,
        source_label: str = "qrng",
    ) -> List[AnomalyEvent]:
        """
        Evaluate all triggers against the latest frames.
        Returns list of fired AnomalyEvents this call.
        """
        fired: List[AnomalyEvent] = []
        self._frame_index += 1

        # Cache PRNG window for KL divergence
        if prng_frame is not None:
            self._prng_windows.append(prng_frame.raw_window)
            if len(self._prng_windows) > 20:
                self._prng_windows.pop(0)

        checks = [
            ("entropy_drop",      qrng_frame.entropy,
             lambda v, t: v < t,
             f"Entropy={qrng_frame.entropy:.4f} below {{}}", "qrng"),

            ("entropy_surge",     qrng_frame.entropy,
             lambda v, t: v > t,
             f"Entropy={qrng_frame.entropy:.4f} above {{}}", "qrng"),

            ("bias_drift",        abs(qrng_frame.bias),
             lambda v, t: v > t,
             f"|bias|={abs(qrng_frame.bias):.4f} above {{}}", "qrng"),

            ("autocorr_spike",    abs(qrng_frame.autocorr_lag1),
             lambda v, t: v > t,
             f"|AC1|={abs(qrng_frame.autocorr_lag1):.4f} above {{}}", "qrng"),

            ("runs_anomaly",      abs(qrng_frame.runs_z),
             lambda v, t: v > t,
             f"|runs_z|={abs(qrng_frame.runs_z):.2f} above {{}}", "qrng"),

            ("hurst_persist",     qrng_frame.hurst,
             lambda v, t: v > t,
             f"Hurst={qrng_frame.hurst:.4f} above {{}}", "qrng"),

            ("hurst_antipersist", qrng_frame.hurst,
             lambda v, t: v < t,
             f"Hurst={qrng_frame.hurst:.4f} below {{}}", "qrng"),

            ("spectral_narrow",   qrng_frame.spectral_flat,
             lambda v, t: v < t,
             f"SpectralFlat={qrng_frame.spectral_flat:.4f} below {{}}", "qrng"),
        ]

        for name, value, condition, msg_template, src in checks:
            event = self._check(name, value, condition, msg_template, src)
            if event:
                fired.append(event)

        # Source divergence (requires both frames)
        if prng_frame is not None:
            kl = self._kl_divergence(qrng_frame.raw_window, prng_frame.raw_window)
            event = self._check(
                "source_diverge", kl,
                lambda v, t: v > t,
                f"KL(QRNG‖PRNG)={kl:.5f} above {{}}",
                "divergence",
            )
            if event:
                event.extra["kl_divergence"] = kl
                fired.append(event)

        # Regime change via composite score
        score = self._composite_score(qrng_frame)
        self._score_history.append(score)
        if len(self._score_history) > 2:
            delta = abs(self._score_history[-1] - self._score_history[-2])
            event = self._check(
                "regime_change", delta,
                lambda v, t: v > t,
                f"Composite score jump={delta:.4f} above {{}}",
                source_label,
            )
            if event:
                event.extra["score_before"] = self._score_history[-2]
                event.extra["score_after"]  = self._score_history[-1]
                fired.append(event)

        return fired

    def close(self) -> None:
        self._csv.close()
        logger.info("Log closed: %s", self._log_path)

    def summary(self) -> Dict[str, int]:
        """Return fire counts per trigger."""
        # recompute from log is expensive; track inline instead
        return dict(self._fire_counts) if hasattr(self, "_fire_counts") else {}

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    def _check(
        self,
        name: str,
        value: float,
        condition: Callable,
        msg_template: str,
        source: str,
    ) -> Optional[AnomalyEvent]:
        cfg = self._triggers.get(name)
        if cfg is None or not cfg.enabled:
            return None

        now = time.time()

        if condition(value, cfg.threshold):
            self._consecutive[name] += 1
        else:
            self._consecutive[name] = 0
            return None

        if self._consecutive[name] < cfg.consecutive:
            return None

        if now - self._last_fire[name] < cfg.cooldown:
            return None

        self._last_fire[name] = now
        msg = msg_template.format(cfg.threshold)

        event = AnomalyEvent(
            trigger_name=name,
            severity=cfg.severity,
            source=source,
            frame_index=self._frame_index,
            timestamp=now,
            feature_value=value,
            threshold=cfg.threshold,
            message=msg,
        )

        self._log_event(event)
        if self._audio:
            _play_event(cfg.audio_event, cfg.severity)
        if self._callback:
            self._callback(event)

        logger.info("* %-8s | %-20s | %s", cfg.severity.upper(), name, msg)
        return event

    @staticmethod
    def _kl_divergence(p_samples: np.ndarray, q_samples: np.ndarray) -> float:
        """Symmetric KL divergence between empirical uint8 histograms."""
        bins = 64  # coarser than 256 for stability with small windows
        p_counts = np.bincount(
            (p_samples.astype(float) * bins / 256).astype(int).clip(0, bins - 1),
            minlength=bins,
        ).astype(float)
        q_counts = np.bincount(
            (q_samples.astype(float) * bins / 256).astype(int).clip(0, bins - 1),
            minlength=bins,
        ).astype(float)
        eps = 1e-10
        p = (p_counts + eps) / (p_counts.sum() + eps * bins)
        q = (q_counts + eps) / (q_counts.sum() + eps * bins)
        kl_pq = np.sum(p * np.log(p / q))
        kl_qp = np.sum(q * np.log(q / p))
        return float((kl_pq + kl_qp) / 2)  # symmetric

    @staticmethod
    def _composite_score(f: FeatureFrame) -> float:
        """Single scalar summarising current stream state."""
        return (
            0.35 * f.entropy
            + 0.20 * (1 - abs(f.bias))
            + 0.15 * f.spectral_flat
            + 0.15 * (1 - abs(f.autocorr_lag1))
            + 0.15 * (1 - abs(f.hurst - 0.5) * 2)  # 1 at H=0.5
        )

    def _log_event(self, e: AnomalyEvent) -> None:
        self._writer.writerow([
            time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(e.timestamp)),
            e.frame_index,
            e.trigger_name,
            e.severity,
            e.source,
            f"{e.feature_value:.6f}",
            f"{e.threshold:.6f}",
            e.message,
        ])
        self._csv.flush()
