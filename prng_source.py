"""
prng_source.py — NumPy PRNG stream that mirrors the QRNGIngestor interface.

Drop-in replacement for comparative experiments.  Produces uint8 samples
from one of four generators so you can compare:

    ANU QRNG   (true quantum)
    MT19937    (standard Mersenne Twister)
    PCG64      (modern PCG — very high statistical quality)
    SFC64      (Small Fast Counting — chaotic, no guarantees)

The identical queue/batch API means the rest of the pipeline is unchanged.
"""

from __future__ import annotations
import threading
import queue
import time
import numpy as np
from typing import Optional, Literal

GeneratorName = Literal["mt19937", "pcg64", "sfc64"]

QUEUE_MAX = 32


class PRNGSource:
    """
    Generates uint8 batches from a NumPy BitGenerator at full speed
    and feeds the same queue interface as QRNGIngestor.

    Parameters
    ----------
    generator : str
        One of "mt19937", "pcg64", "sfc64".
    batch_size : int
        Samples per batch (match QRNGIngestor.batch_size for fair comparison).
    poll_interval : float
        Pause between batches — set equal to QRNG poll_interval so both
        sources advance at the same rate.
    seed : int or None
        RNG seed.  None = random seed (non-reproducible).
    verbose : bool
    """

    GENERATORS = {
        "mt19937": np.random.MT19937,
        "pcg64":   np.random.PCG64,
        "sfc64":   np.random.SFC64,
    }

    def __init__(
        self,
        generator: GeneratorName = "pcg64",
        batch_size: int = 1024,
        poll_interval: float = 0.5,
        seed: Optional[int] = None,
        verbose: bool = True,
    ):
        if generator not in self.GENERATORS:
            raise ValueError(f"Unknown generator '{generator}'. Choose from {list(self.GENERATORS)}")

        self.generator_name = generator
        self.batch_size = batch_size
        self.poll_interval = poll_interval
        self.verbose = verbose

        bit_gen = self.GENERATORS[generator](seed=seed)
        self._rng = np.random.Generator(bit_gen)
        self._queue: queue.Queue = queue.Queue(maxsize=QUEUE_MAX)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------ #
    #  Public API  (identical to QRNGIngestor)                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._gen_loop, daemon=True)
        self._thread.start()
        if self.verbose:
            print(f"[prng] {self.generator_name.upper()} source started  "
                  f"(batch={self.batch_size}).")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
        if self.verbose:
            print(f"[prng] {self.generator_name.upper()} source stopped.")

    def get_batch(self, timeout: float = 5.0) -> Optional[np.ndarray]:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def queue_depth(self) -> int:
        return self._queue.qsize()

    # ------------------------------------------------------------------ #
    #  Internal                                                            #
    # ------------------------------------------------------------------ #

    def _gen_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._queue.qsize() >= QUEUE_MAX - 2:
                time.sleep(self.poll_interval)
                continue
            batch = self._rng.integers(0, 256, size=self.batch_size, dtype=np.uint8)
            self._queue.put(batch)
            time.sleep(self.poll_interval)
