"""
features.py — Rolling window feature extraction for QRNG streams.

Each window of uint8 samples produces a FeatureFrame:
    entropy          Shannon entropy (bits), normalised to [0,1] over max 8 bits
    bias             Signed deviation of mean from 127.5, normalised [-1,1]
    autocorr_lag1    Pearson autocorrelation at lag 1
    runs_z           Runs-test z-score (signed)
    hurst            Hurst exponent via R/S analysis
    spectral_flat    Wiener spectral flatness of sample histogram
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Deque
from collections import deque
import scipy.stats as stats


MAX_ENTROPY_BITS = 8.0  # uint8 theoretical max


@dataclass
class FeatureFrame:
    """One window's worth of extracted features, all normalised to [-1, 1] or [0, 1]."""

    raw_window: np.ndarray          # original uint8 samples
    entropy: float = 0.0            # [0, 1]
    bias: float = 0.0               # [-1, 1]  positive = skewed high
    autocorr_lag1: float = 0.0      # [-1, 1]
    runs_z: float = 0.0             # unbounded signed z-score, clipped to [-5,5]
    hurst: float = 0.5              # [0, 1]  0.5 = random, >0.5 = persistent
    spectral_flat: float = 1.0      # [0, 1]  1.0 = perfectly flat (white noise)
    
    # Additional statistical tests
    chi_square: float = 0.0         # Chi-square goodness of fit to uniform
    serial_corr: float = 0.0        # Serial correlation at lag > 1
    gap_test_z: float = 0.0         # Gap test for randomness
    timestamp: float = 0.0


class FeatureEngine:
    """
    Maintains a rolling deque of uint8 samples and emits FeatureFrames
    every `step` samples.

    Parameters
    ----------
    window : int
        Window size in samples.
    step : int
        How many new samples trigger a new frame (overlap = window - step).
    history : int
        How many FeatureFrames to keep in memory.
    """

    def __init__(self, window: int = 512, step: int = 128, history: int = 200):
        self.window = window
        self.step = step
        self._buffer: Deque[int] = deque(maxlen=window)
        self._step_counter = 0
        self._frames: Deque[FeatureFrame] = deque(maxlen=history)
        self._time = 0.0

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def push(self, samples: np.ndarray) -> list[FeatureFrame]:
        """
        Feed new samples into the engine.
        Returns a (possibly empty) list of newly computed FeatureFrames.
        """
        new_frames = []
        for s in samples.astype(np.uint8):
            self._buffer.append(int(s))
            self._step_counter += 1
            self._time += 1.0

            if (
                self._step_counter >= self.step
                and len(self._buffer) == self.window
            ):
                frame = self._compute(np.array(self._buffer, dtype=np.float64))
                frame.timestamp = self._time
                self._frames.append(frame)
                new_frames.append(frame)
                self._step_counter = 0

        return new_frames

    @property
    def frames(self) -> list[FeatureFrame]:
        return list(self._frames)

    # ------------------------------------------------------------------ #
    #  Feature computation                                                 #
    # ------------------------------------------------------------------ #

    def _compute(self, w: np.ndarray) -> FeatureFrame:
        frame = FeatureFrame(raw_window=w.astype(np.uint8))
        frame.entropy = self._entropy(w)
        frame.bias = self._bias(w)
        frame.autocorr_lag1 = self._autocorr(w, lag=1)
        frame.runs_z = self._runs_z(w)
        frame.hurst = self._hurst(w)
        frame.spectral_flat = self._spectral_flatness(w)
        
        # Additional statistical tests
        frame.chi_square = self._chi_square(w)
        frame.serial_corr = self._serial_correlation(w, lag=5)
        frame.gap_test_z = self._gap_test(w)
        
        return frame

    @staticmethod
    def _entropy(w: np.ndarray) -> float:
        counts = np.bincount(w.astype(np.int64), minlength=256)
        p = counts / counts.sum()
        p = p[p > 0]
        h = -np.sum(p * np.log2(p))
        return float(np.clip(h / MAX_ENTROPY_BITS, 0.0, 1.0))

    @staticmethod
    def _bias(w: np.ndarray) -> float:
        mean = w.mean()
        deviation = (mean - 127.5) / 127.5  # normalised to [-1, 1]
        return float(np.clip(deviation, -1.0, 1.0))

    @staticmethod
    def _autocorr(w: np.ndarray, lag: int = 1) -> float:
        if len(w) <= lag:
            return 0.0
        x = w - w.mean()
        denom = np.dot(x, x)
        if denom == 0:
            return 0.0
        return float(np.clip(np.dot(x[:-lag], x[lag:]) / denom, -1.0, 1.0))

    @staticmethod
    def _runs_z(w: np.ndarray) -> float:
        median = np.median(w)
        above = (w > median).astype(int)
        runs = 1 + np.sum(np.diff(above) != 0)
        n1 = np.sum(above)
        n2 = len(w) - n1
        n = n1 + n2
        if n == 0 or n1 == 0 or n2 == 0:
            return 0.0
        expected = (2 * n1 * n2) / n + 1
        variance = (2 * n1 * n2 * (2 * n1 * n2 - n)) / (n ** 2 * (n - 1))
        if variance <= 0:
            return 0.0
        z = (runs - expected) / np.sqrt(variance)
        return float(np.clip(z, -5.0, 5.0))

    @staticmethod
    def _hurst(w: np.ndarray) -> float:
        """R/S analysis Hurst exponent estimate."""
        n = len(w)
        if n < 20:
            return 0.5
        try:
            ts = w - w.mean()
            cumdev = np.cumsum(ts)
            R = cumdev.max() - cumdev.min()
            S = w.std()
            if S == 0:
                return 0.5
            RS = R / S
            if RS <= 0:
                return 0.5
            H = np.log(RS) / np.log(n / 2.0)
            return float(np.clip(H, 0.0, 1.0))
        except Exception:
            return 0.5

    @staticmethod
    def _spectral_flatness(w: np.ndarray) -> float:
        """Wiener spectral flatness of the value histogram."""
        counts = np.bincount(w.astype(np.int64), minlength=256).astype(float)
        counts += 1e-10  # avoid log(0)
        geometric_mean = np.exp(np.mean(np.log(counts)))
        arithmetic_mean = counts.mean()
        if arithmetic_mean == 0:
            return 0.0
        return float(np.clip(geometric_mean / arithmetic_mean, 0.0, 1.0))

    @staticmethod
    def _chi_square(w: np.ndarray) -> float:
        """
        Chi-square goodness of fit test to uniform distribution.
        Returns z-score normalized value where higher = more deviation from uniform.
        """
        counts = np.bincount(w.astype(np.int64), minlength=256).astype(float)
        n = counts.sum()
        expected = n / 256.0
        
        if expected == 0:
            return 0.0
            
        chi_sq = np.sum((counts - expected) ** 2 / expected)
        
        # Degrees of freedom = 255 (number of bins - 1)
        df = 255
        
        # Approximate using normal distribution for large df
        # Mean of chi-square = df, Variance = 2*df
        mean_chi_sq = df
        std_chi_sq = np.sqrt(2 * df)
        
        z_score = (chi_sq - mean_chi_sq) / std_chi_sq
        
        # Normalize to [0,1] range using sigmoid-like transformation
        normalized = np.tanh(z_score / 5.0) / 2.0 + 0.5
        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def _serial_correlation(w: np.ndarray, lag: int = 5) -> float:
        """
        Serial correlation at specified lag.
        Measures correlation between w[i] and w[i+lag].
        Returns value in [-1, 1].
        """
        if len(w) <= lag:
            return 0.0
            
        x = w - w.mean()
        
        # Correlation with lag offset
        numerator = np.dot(x[:-lag], x[lag:])
        denominator = np.dot(x, x)
        
        if denominator == 0:
            return 0.0
        
        corr = numerator / denominator
        return float(np.clip(corr, -1.0, 1.0))

    @staticmethod
    def _gap_test(w: np.ndarray) -> float:
        """
        Gap test for randomness.
        Counts gaps between values in a specified range (e.g., 120-135).
        Returns z-score indicating deviation from expected gap distribution.
        """
        # Define target range (middle of uint8 range)
        lower, upper = 120, 135
        n_total = len(w)
        
        # Count values in target range
        in_range = np.sum((w >= lower) & (w <= upper))
        n_target = int(in_range)
        
        if n_target < 3:
            return 0.0
            
        # Probability of being in target range
        p = (upper - lower + 1) / 256.0
        
        # Find gaps between consecutive values in target range
        indices = np.where((w >= lower) & (w <= upper))[0]
        
        if len(indices) < 2:
            return 0.0
            
        # Gap sizes (distance minus one)
        gaps = np.diff(indices) - 1
        
        n_gaps = len(gaps)
        
        # Expected mean gap size for geometric distribution: (1-p)/p
        expected_mean_gap = (1 - p) / p
        variance = (1 - p) / (p ** 2)
        
        if variance <= 0 or n_gaps < 5:
            return 0.0
            
        observed_mean = gaps.mean()
        z_score = (observed_mean - expected_mean_gap) / np.sqrt(variance / n_gaps)
        
        # Clip to reasonable range and normalize
        clipped_z = np.clip(z_score, -5.0, 5.0)
        normalized = (clipped_z + 5.0) / 10.0
        
        return float(np.clip(normalized, 0.0, 1.0))

    @staticmethod
    def _autocorr_lag(k: int, w: np.ndarray) -> float:
        """
        Autocorrelation at lag k (generalization of autocorr_lag1).
        
        Parameters
        ----------
        k : int
            Lag value.
        w : ndarray
            Input array.
            
        Returns
        -------
        float
            Autocorrelation at lag k, in [-1, 1].
        """
        if len(w) <= k:
            return 0.0
            
        x = w - w.mean()
        denom = np.dot(x, x)
        
        if denom == 0:
            return 0.0
            
        ac = np.dot(x[:-k], x[k:]) / denom
        return float(np.clip(ac, -1.0, 1.0))

    @staticmethod
    def _autocorr(w: np.ndarray, lag: int = 1) -> float:
        """Alias for autocorr_lag(k)."""
        return FeatureEngine._autocorr_lag(lag, w)
