"""
ingest.py — ANU QRNG API stream ingestor with improved error handling.

Fetches uint8 samples in batches and feeds a thread-safe queue.
Includes retry logic, exponential backoff, and health monitoring.
"""

import threading
import queue
import time
import requests
from requests.exceptions import RequestException, Timeout, ConnectionError, HTTPError
import numpy as np
from typing import Optional, Callable, Tuple

ANU_URL = "https://qrng.anu.edu.au/API/jsonI.php"
BATCH_SIZE = 1024  # uint8 samples per request (API max ~1024)
POLL_INTERVAL = 0.5  # seconds between requests when buffer is healthy
QUEUE_MAX = 32  # max batches buffered

# Retry configuration
DEFAULT_RETRY_MAX_ATTEMPTS = 3
DEFAULT_RETRY_BACKOFF_BASE = 2.0


class QRNGIngestorError(Exception):
    """Exception raised for QRNG ingestor errors."""
    pass


class QRNGHealthMonitor:
    """Monitors the health of the QRNG data stream."""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self._recent_values: list = []

    def record_value(self, value: float) -> None:
        """Record a measurement for health checking."""
        self._recent_values.append(value)
        if len(self._recent_values) > self.window_size:
            self._recent_values.pop(0)

    @property
    def recent_mean(self) -> float:
        """Mean of recent measurements."""
        return sum(self._recent_values) / len(self._recent_values) if self._recent_values else 0.0

    @property
    def is_stable(self) -> bool:
        """Check if stream is producing consistent data."""
        if len(self._recent_values) < 10:
            return False
        # Check for sudden drops in queue depth (potential API issues)
        variance = sum((x - self.recent_mean) ** 2 for x in self._recent_values) / len(self._recent_values)
        return variance < 5.0


class QRNGIngestor:
    """
    Polls the ANU QRNG API continuously and fills a shared sample queue.

    Features:
        - Automatic retry with exponential backoff on failures
        - Health monitoring for stream quality
        - Graceful degradation under adverse conditions

    Parameters
    ----------
    batch_size : int
        Samples per API request (1–1024, uint8).
    poll_interval : float
        Minimum wait between requests in seconds.
    verbose : bool
        Print fetch confirmations to stdout.
    retry_max_attempts : int
        Maximum retry attempts before giving up on a single request.
    retry_backoff_base : float
        Base for exponential backoff calculation.
    health_monitor : QRNGHealthMonitor or None
        Optional health monitor instance.
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        poll_interval: float = POLL_INTERVAL,
        verbose: bool = True,
        retry_max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
        retry_backoff_base: float = DEFAULT_RETRY_BACKOFF_BASE,
        health_monitor: Optional[QRNGHealthMonitor] = None,
    ):
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.verbose = verbose
        
        # Retry configuration
        self._retry_max_attempts = retry_max_attempts
        self._retry_backoff_base = retry_backoff_base

        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        
        # Health tracking
        self._error_count = 0
        self._total_successes = 0
        self._total_failures = 0
        self._health_monitor = health_monitor or QRNGHealthMonitor()

    @property
    def total_successes(self) -> int:
        """Total successful fetch count."""
        return self._total_successes

    @property
    def total_failures(self) -> int:
        """Total failed fetch count."""
        return self._total_failures

    @property
    def error_rate(self) -> float:
        """Current error rate (0.0 to 1.0)."""
        total = self._total_successes + self._total_failures
        if total == 0:
            return 0.0
        return self._total_failures / total

    @property
    def is_healthy(self) -> bool:
        """Check if ingestor is considered healthy."""
        return (self.error_rate < 0.3 and 
                self._health_monitor.is_stable if hasattr(self, '_health_monitor') else True)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Begin background polling thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print("[ingest] ANU QRNG polling started.")

    def stop(self) -> None:
        """Signal the polling thread to stop."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self.verbose:
            print("[ingest] Polling stopped.")

    def get_batch(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        """
        Blocking fetch of one batch (uint8 ndarray, shape [batch_size]).
        Returns None on timeout.
        """
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _poll_loop(self) -> None:
        """Main polling loop that fetches batches from the API."""
        while not self._stop_event.is_set():
            # Don't over-fill the buffer
            if self._queue.qsize() >= QUEUE_MAX - 2:
                time.sleep(self.poll_interval)
                continue

            batch = self._fetch_batch_with_retry()
            
            if batch is not None:
                self._queue.put(batch)
                self._error_count = 0
                self._total_successes += 1
                
                # Update health metrics
                queue_depth = self._queue.qsize()
                self._health_monitor.record_value(queue_depth)
                
                if self.verbose:
                    print(
                        f"[ingest] +{len(batch)} samples  "
                        f"(queue depth: {self._queue.qsize()}, success #{self._total_successes})"
                    )
            else:
                self._error_count += 1
                self._total_failures += 1
                
                # Calculate backoff with exponential increase
                backoff = min(
                    self._retry_backoff_base ** (min(self._error_count, 5)),
                    30.0  # Max backoff of 30 seconds
                )
                
                if self.verbose:
                    print(
                        f"[ingest] Fetch failed (attempt {self._error_count}) - "
                        f"backing off {backoff:.1f}s, error_rate={self.error_rate:.2%}"
                    )
                    
                time.sleep(backoff)

            # Minimum interval between polling attempts
            time.sleep(self.poll_interval)

    def _fetch_batch_with_retry(self) -> Optional[np.ndarray]:
        """Fetch a batch with automatic retry on failure."""
        last_error = None
        
        for attempt in range(1, self._retry_max_attempts + 1):
            try:
                # Add slight jitter to prevent thundering herd
                time.sleep(attempt * 0.1) if attempt > 1 else None
                
                batch = self._fetch_batch()
                if batch is not None:
                    return batch
                    
            except RequestException as e:
                last_error = e
                if isinstance(e, Timeout):
                    if self.verbose:
                        print(f"[ingest] Timeout on attempt {attempt}")
                elif isinstance(e, HTTPError) and e.response.status_code == 429:
                    # Rate limited - wait longer
                    wait_time = min(2 ** attempt * 5, 60)
                    if self.verbose:
                        print(f"[ingest] Rate limited (429), waiting {wait_time}s")
                    time.sleep(wait_time)
                elif isinstance(e, ConnectionError):
                    if self.verbose:
                        print(f"[ingest] Connection error on attempt {attempt}")
            
            # Retry with exponential backoff for non-429 errors
            if attempt < self._retry_max_attempts and last_error:
                backoff = min(
                    self._retry_backoff_base ** attempt,
                    10.0
                )
                time.sleep(backoff)

        # All retries exhausted
        if last_error:
            if self.verbose:
                print(f"[ingest] All {self._retry_max_attempts} attempts failed: {type(last_error).__name__}")
        
        return None

    def _fetch_batch(self, timeout: int = 10) -> Optional[np.ndarray]:
        """
        Fetch a single batch from the API without retry.
        
        Parameters
        ----------
        timeout : int
            Request timeout in seconds.
            
        Returns
        -------
        ndarray or None
            Array of uint8 samples, or None on failure.
        """
        try:
            params = {
                "length": self.batch_size,
                "type": "uint8",
            }
            resp = requests.get(ANU_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            
            data = resp.json()
            if data.get("success"):
                return np.array(data["data"], dtype=np.uint8)
            else:
                if self.verbose:
                    print(f"[ingest] API returned success=False")
                return None
                
        except Timeout:
            raise Timeout("API request timed out")
            
        except HTTPError as e:
            # 429 = rate limited, other errors are permanent failures for this attempt
            if e.response.status_code == 429:
                raise e
            else:
                raise HTTPError(f"HTTP error {e.response.status_code}")
                
        except RequestException as e:
            raise
            
        except Exception as e:
            if self.verbose:
                print(f"[ingest] Unexpected error: {type(e).__name__}: {e}")
            return None

    def get_stats(self) -> Tuple[int, int, float]:
        """Return (total_successes, total_failures, error_rate)."""
        return (self._total_successes, self._total_failures, self.error_rate)
