"""
sonifier.py — Real-time audio synthesis from FeatureFrames.

Synthesis model: additive FM with parameter smoothing.

Single-source feature -> audio mapping
──────────────────────────────────────
entropy          → amplitude           (high entropy = louder)
bias             → pitch center offset  (biased high = sharper)
autocorr_lag1    → harmonic richness    (positive AC = more partials)
runs_z           → stereo width         (high |z| = wider)
hurst            → LFO rate             (persistent = slower modulation)
spectral_flat    → brightness filter    (flat = brighter)

Dual-source spatial layout (when PRNG comparison is active)
────────────────────────────────────────────────────────────
  QRNG  →  panned LEFT   (tuned to A3 = 220 Hz)
  PRNG  ->  panned RIGHT  (tuned to E3 = 165 Hz - a fifth below)
  KL divergence between sources -> centre sub-oscillator amplitude
"""

from __future__ import annotations
import numpy as np
import threading
import time
from typing import Optional
from features import FeatureFrame

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("[sonifier] Warning: sounddevice not installed. Audio output disabled.")


SAMPLE_RATE = 44100
BLOCK_SIZE = 1024           # frames per audio callback
BASE_FREQ = 220.0           # A3 - QRNG centre pitch
PRNG_FREQ = 165.0           # E3 - PRNG centre pitch (perfect fifth below)
PITCH_RANGE_CENTS = 400     # +/-200 cents (~two semitones) for bias modulation
SMOOTH_ALPHA = 0.05         # low-pass smoothing for parameter transitions


class SynthState:
    """Mutable synthesis parameters, updated from feature frames."""

    def __init__(self, base_freq: float = BASE_FREQ):
        self._base_freq = base_freq
        self.amplitude = 0.5
        self.pitch_hz = base_freq
        self.n_harmonics = 3
        self.lfo_rate_hz = 0.5
        self.stereo_width = 0.3
        self.brightness = 0.8

        # Phase accumulators (per harmonic)
        self._phases = np.zeros(16)
        self._lfo_phase = 0.0

        self._lock = threading.Lock()

    def update_from_frame(self, frame: FeatureFrame) -> None:
        """Translate feature values into synthesis parameters (with smoothing)."""
        with self._lock:
            target_amp = 0.2 + 0.6 * frame.entropy
            target_pitch = self._base_freq * (
                2 ** ((frame.bias * PITCH_RANGE_CENTS / 2) / 1200.0)
            )
            # autocorr_lag1 in [-1,1] -> harmonics [1,8]
            target_harmonics = int(1 + 7 * np.clip((frame.autocorr_lag1 + 1) / 2, 0, 1))
            target_lfo = 0.1 + 2.0 * frame.hurst          # 0.1–2.1 Hz
            target_width = np.abs(frame.runs_z) / 5.0      # 0–1
            target_brightness = frame.spectral_flat         # 0–1

            # Apply smoothing
            a = SMOOTH_ALPHA
            self.amplitude = (1 - a) * self.amplitude + a * target_amp
            self.pitch_hz = (1 - a) * self.pitch_hz + a * target_pitch
            self.n_harmonics = max(1, round((1 - a) * self.n_harmonics + a * target_harmonics))
            self.lfo_rate_hz = (1 - a) * self.lfo_rate_hz + a * target_lfo
            self.stereo_width = (1 - a) * self.stereo_width + a * target_width
            self.brightness = (1 - a) * self.brightness + a * target_brightness


class Sonifier:
    """
    Real-time additive FM synthesiser driven by FeatureFrames.

    Single-source:
        s = Sonifier()
        s.start()
        s.push_frame(qrng_frame)

    Dual-source (QRNG left / PRNG right):
        s = Sonifier(dual_source=True)
        s.start()
        s.push_frame(qrng_frame, prng_frame=prng_frame, kl_divergence=kl)
    """

    def __init__(self, sample_rate: int = SAMPLE_RATE, dual_source: bool = False):
        self.sample_rate = sample_rate
        self.dual_source = dual_source
        self._qrng_state = SynthState(base_freq=BASE_FREQ)
        self._prng_state = SynthState(base_freq=PRNG_FREQ) if dual_source else None
        self._divergence_amp = 0.0   # sub-oscillator amplitude driven by KL
        self._stream: Optional[object] = None
        self._running = False
        self._frame_count = 0

    def start(self) -> None:
        if not SOUNDDEVICE_AVAILABLE:
            print("[sonifier] sounddevice unavailable - audio disabled.")
            return
        self._running = True
        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=2,
            blocksize=BLOCK_SIZE,
            dtype="float32",
            callback=self._audio_callback,
        )
        self._stream.start()
        mode = "dual-source" if self.dual_source else "single-source"
        print(f"[sonifier] Audio stream started ({mode}).")

    def stop(self) -> None:
        self._running = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
        print("[sonifier] Audio stream stopped.")

    def push_frame(
        self,
        qrng_frame: FeatureFrame,
        prng_frame: Optional[FeatureFrame] = None,
        kl_divergence: float = 0.0,
    ) -> None:
        """Update synthesis parameters from new feature frames."""
        self._qrng_state.update_from_frame(qrng_frame)
        if self._prng_state is not None and prng_frame is not None:
            self._prng_state.update_from_frame(prng_frame)
        # Smooth KL into divergence amplitude (0–0.3 audible range)
        a = SMOOTH_ALPHA
        self._divergence_amp = (
            (1 - a) * self._divergence_amp
            + a * float(np.clip(kl_divergence * 6, 0.0, 0.3))
        )
        self._frame_count += 1

    # ------------------------------------------------------------------ #
    #  Audio callback (runs on dedicated audio thread)                    #
    # ------------------------------------------------------------------ #

    def _audio_callback(self, outdata: np.ndarray, frames: int, time_info, status):
        left  = self._render_voice(self._qrng_state, frames)
        right = self._render_voice(
            self._prng_state if self._prng_state else self._qrng_state, frames
        )

        if self.dual_source:
            # Hard pan: QRNG -> left, PRNG -> right
            # Centre sub-oscillator carries KL divergence signal
            sub = self._render_sub(frames, self._divergence_amp)
            left  = left  * 0.8 + sub * 0.5
            right = right * 0.8 + sub * 0.5
        else:
            # Single source: use stereo width from runs_z
            with self._qrng_state._lock:
                width = self._qrng_state.stereo_width
            mid  = (left + right) / 2
            side = (left - right) / 2
            left  = mid + side * (1 + width)
            right = mid - side * (1 + width)

        # Normalise to prevent clipping
        peak = max(np.abs(left).max(), np.abs(right).max(), 1e-6)
        if peak > 0.9:
            left  /= peak / 0.9
            right /= peak / 0.9

        outdata[:, 0] = left.astype(np.float32)
        outdata[:, 1] = right.astype(np.float32)

    def _render_voice(self, s: SynthState, frames: int) -> np.ndarray:
        """Render one additive FM voice, return mono float64 array."""
        t = np.arange(frames) / self.sample_rate
        with s._lock:
            amp        = float(s.amplitude)
            freq       = float(s.pitch_hz)
            n_h        = int(s.n_harmonics)
            lfo_rate   = float(s.lfo_rate_hz)
            brightness = float(s.brightness)

        lfo = 1.0 + 0.15 * np.sin(2 * np.pi * lfo_rate * t + s._lfo_phase)
        s._lfo_phase = (s._lfo_phase + 2 * np.pi * lfo_rate * frames / self.sample_rate) % (2 * np.pi)

        mono = np.zeros(frames, dtype=np.float64)
        for h in range(1, n_h + 1):
            hf = freq * h
            if hf > self.sample_rate / 2:
                break
            gain = (brightness ** (h - 1)) / h
            phase_inc = 2 * np.pi * hf / self.sample_rate
            phases = s._phases[h - 1] + phase_inc * np.arange(frames)
            mono += gain * np.sin(phases)
            s._phases[h - 1] = phases[-1] % (2 * np.pi)

        return mono * amp * lfo

    def _render_sub(self, frames: int, amp: float) -> np.ndarray:
        """Centre sub-oscillator - amplitude tracks KL divergence."""
        if not hasattr(self, "_sub_phase"):
            self._sub_phase = 0.0
        freq = 55.0  # A1 - deep rumble
        phase_inc = 2 * np.pi * freq / self.sample_rate
        phases = self._sub_phase + phase_inc * np.arange(frames)
        self._sub_phase = phases[-1] % (2 * np.pi)
        return np.sin(phases) * amp
