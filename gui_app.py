"""
gui_app.py — Tkinter-based GUI for QRNG Sonifier.

Provides a desktop application with:
- Real-time feature visualization (matplotlib embedded)
- Live spectrogram display
- Anomaly event log
- Control buttons and configuration panel
"""

from __future__ import annotations
import logging
import threading
import time
from typing import Optional

import tkinter as tk
from tkinter import ttk, messagebox

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

from features import FeatureEngine, FeatureFrame
from ingest import QRNGIngestor

logger = logging.getLogger(__name__)


class QRNGSonifierGUI:
    """Desktop GUI application for QRNG Sonifier."""

    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("QRNG Sonifier")
        self.root.geometry("1200x800")

        # State variables
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frames: list[FeatureFrame] = []
        self._frame_counter = 0  # monotonically increasing frame index for x-axis
        self._anomalies: list[dict] = []
        self._start_time = 0.0

        # Matplotlib figure
        self.fig, (self.ax_main, self.ax_spectrogram) = plt.subplots(2, 1, figsize=(10, 6), dpi=100)

        # Setup UI
        self._setup_ui()

    def _setup_ui(self) -> None:
        """Set up the GUI layout."""

        # Top control panel
        control_frame = ttk.Frame(self.root, padding="10")
        control_frame.pack(fill=tk.X)

        self.btn_start = ttk.Button(control_frame, text="Start", command=self._start)
        self.btn_start.pack(side=tk.LEFT, padx=5)

        self.btn_stop = ttk.Button(control_frame, text="Stop", command=self._stop, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=5)

        ttk.Label(control_frame, text="Status:").pack(side=tk.LEFT, padx=(20, 5))
        self.lbl_status = ttk.Label(control_frame, text="Stopped", foreground="red")
        self.lbl_status.pack(side=tk.LEFT, padx=5)

        # Main visualization area
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(canvas, main_frame)
        toolbar.update()

        # Bottom panel — statistics and anomalies
        bottom_frame = ttk.Frame(self.root, padding="10")
        bottom_frame.pack(fill=tk.X)

        stats_frame = ttk.LabelFrame(bottom_frame, text="Live Statistics", padding="5")
        stats_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)

        self.lbl_entropy = ttk.Label(stats_frame, text="Entropy: --")
        self.lbl_entropy.pack()

        self.lbl_bias = ttk.Label(stats_frame, text="Bias: --")
        self.lbl_bias.pack()

        self.lbl_hurst = ttk.Label(stats_frame, text="Hurst: --")
        self.lbl_hurst.pack()

        self.lbl_chisq = ttk.Label(stats_frame, text="Chi-Square: --")
        self.lbl_chisq.pack()

        anomaly_frame = ttk.LabelFrame(bottom_frame, text="Anomalies", padding="5")
        anomaly_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=False)

        self.txt_anomalies = tk.Text(anomaly_frame, height=8, width=40)
        self.txt_anomalies.pack(fill=tk.BOTH)

        self._init_plots()

    def _init_plots(self) -> None:
        """Initialize matplotlib plots."""
        self.ax_main.set_title('Feature Timeline')
        self.ax_main.set_xlabel('Frame Index')
        self.ax_main.set_ylabel('Value')
        self.ax_main.grid(True, alpha=0.3)

        colors = {
            'entropy': 'orange',
            'bias': 'red',
            'hurst': 'green',
            'spectral_flat': 'blue',
            'chi_square': 'purple',
        }
        self.lines: dict = {}
        for name, color in colors.items():
            self.lines[name], = self.ax_main.plot([], [], label=name.capitalize(), color=color)

        self.ax_main.legend(loc='upper right')
        self.ax_main.set_ylim(-1.2, 1.2)

        self.ax_spectrogram.set_title('Spectrogram (placeholder)')
        self.ax_spectrogram.set_xlabel('Sample Index')
        self.ax_spectrogram.set_ylabel('Frequency Bin')

    def _start(self) -> None:
        """Start the QRNG sonifier."""
        if self._running:
            return

        self._running = True
        self._frames.clear()
        self._frame_counter = 0
        self._anomalies.clear()
        self._start_time = time.time()

        self.btn_start.config(state=tk.DISABLED)
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text="Running", foreground="green")

        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _stop(self) -> None:
        """Stop the QRNG sonifier."""
        self._running = False

        self.btn_start.config(state=tk.NORMAL)
        self.btn_stop.config(state=tk.DISABLED)
        self.lbl_status.config(text="Stopped", foreground="red")

        if self._thread:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        """Main loop — fetches real QRNG data and drives feature extraction."""
        engine = FeatureEngine(window=512, step=128, history=200)
        ingestor = QRNGIngestor(batch_size=256, poll_interval=0.5, verbose=False)
        ingestor.start()

        try:
            while self._running:
                batch = ingestor.get_batch(timeout=5.0)
                if batch is None:
                    logger.warning("GUI: timeout waiting for QRNG batch — retrying.")
                    continue

                new_frames = engine.push(batch)

                for frame in new_frames:
                    self._frames.append(frame)
                    self._frame_counter += 1

                    self.root.after(0, lambda f=frame: self._update_stats(f))
                    self._check_anomaly(frame)

                if self._frame_counter % 5 == 0:
                    self.root.after(0, self._update_plot)

        except Exception as e:
            logger.exception("Unexpected error in GUI run loop: %s", e)
        finally:
            ingestor.stop()

    def _check_anomaly(self, frame: FeatureFrame) -> None:
        """Fire an anomaly entry when entropy drops below threshold."""
        if frame.entropy >= 0.85:
            return

        # Rate-limit: do not fire within 30 s of the last anomaly
        if self._anomalies and (time.time() - self._anomalies[-1]['time'] < 30):
            return

        anomaly = {
            'trigger': 'entropy_drop',
            'severity': 'warning',
            'value': frame.entropy,
            'time': time.time(),
        }
        self._anomalies.append(anomaly)
        self.root.after(0, lambda a=anomaly: self._log_anomaly(a))

    def _update_stats(self, frame: FeatureFrame) -> None:
        """Update statistics labels."""
        self.lbl_entropy.config(text=f"Entropy: {frame.entropy:.3f}")
        self.lbl_bias.config(text=f"Bias: {frame.bias:.3f}")
        self.lbl_hurst.config(text=f"Hurst: {frame.hurst:.3f}")
        self.lbl_chisq.config(text=f"Chi-Square: {frame.chi_square:.3f}")

    def _log_anomaly(self, anomaly: dict) -> None:
        """Append an anomaly event to the text log."""
        severity = anomaly['severity'].upper()
        msg = f"[{severity}] {anomaly['trigger']}: {anomaly['value']:.3f}\n"
        self.txt_anomalies.insert(tk.END, msg)
        self.txt_anomalies.see(tk.END)

    def _update_plot(self) -> None:
        """Refresh the matplotlib feature timeline plot."""
        if not self._frames:
            return

        window = self._frames[-100:]
        # Build x-axis indices from the global frame counter
        start_idx = self._frame_counter - len(window)
        indices = list(range(start_idx, self._frame_counter))

        for name, line in self.lines.items():
            values = [getattr(f, name, 0.0) for f in window]
            line.set_data(indices, values)

        self.ax_main.relim()
        self.ax_main.autoscale_view()
        self.fig.tight_layout()
        self.fig.canvas.draw_idle()

    def run(self) -> None:
        """Start the GUI main loop."""
        self.root.mainloop()


def main() -> None:
    """Entry point for GUI application."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    app = QRNGSonifierGUI()
    app.run()


if __name__ == '__main__':
    main()
