"""
main.py — QRNG Sonifier — research demo entry point.

Usage
─────
  python main.py                                # QRNG only, all features on
  python main.py --prng                         # dual-source: QRNG + PCG64 PRNG
  python main.py --prng --prng-gen mt19937      # use Mersenne Twister instead
  python main.py --duration 300                 # run for 5 minutes then export
  python main.py --no-audio                     # metrics + spectrogram, no sound
  python main.py --no-anomaly                   # disable anomaly trigger layer
  python main.py --render-interval 30           # export PNG every 30s
  python main.py --output-dir ./session1        # custom output directory
  python main.py --window 256 --step 64         # finer time resolution

Outputs (in --output-dir)
─────────────────────────
  metrics_<ts>.csv                one row per frame, all six features
  anomaly_log_<ts>.csv            every fired trigger event (if enabled)
  spectrogram_<ts>_frame*.png     periodic dual-panel spectrogram
  spectrogram_<ts>_final.png      export on exit

Dependencies
────────────
  pip install requests numpy scipy matplotlib sounddevice
  # Linux: sudo apt install portaudio19-dev
"""

from __future__ import annotations
import argparse
import logging
import signal
import time
import yaml
from pathlib import Path
from ingest import QRNGIngestor
from prng_source import PRNGSource
from features import FeatureEngine
from sonifier import Sonifier
from renderer import Renderer
from anomaly_triggers import AnomalyDetector

logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file, deep-merging with defaults."""
    defaults: dict = {
        "output_dir": "./output",
        "duration": None,
        "render_interval": 60.0,
        "prng_comparison": {"enabled": False},
        "batch_size": 1024,
        "window": 512,
        "step": 128,
    }

    config = defaults.copy()

    if Path(config_path).exists():
        with open(config_path, 'r') as f:
            file_config = yaml.safe_load(f) or {}

        for key, value in file_config.items():
            if isinstance(value, dict):
                config[key] = {**defaults.get(key, {}), **value}
            else:
                config[key] = value
    else:
        logger.warning("Config file not found: %s — using defaults.", config_path)

    return config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="QRNG Sonifier - real-time audio + spectrogram + metrics"
    )
    # Use None as sentinel so we can detect "not explicitly set" vs "default"
    p.add_argument("--duration",          type=float, default=None)
    p.add_argument("--output-dir",        type=str,   default=None)
    p.add_argument("--render-interval",   type=float, default=None)
    p.add_argument("--prng",              action="store_true",
                   help="Enable PRNG comparison stream (dual-source mode)")
    p.add_argument("--prng-gen",          type=str,   default=None,
                   choices=["mt19937", "pcg64", "sfc64"])
    p.add_argument("--prng-seed",         type=int,   default=None)
    p.add_argument("--batch-size",        type=int,   default=None)
    p.add_argument("--window",            type=int,   default=None)
    p.add_argument("--step",              type=int,   default=None)
    p.add_argument("--no-audio",          action="store_true")
    p.add_argument("--no-anomaly",        action="store_true")
    p.add_argument("--anomaly-no-audio",  action="store_true",
                   help="Log anomalies but suppress audio events")
    p.add_argument("--verbose",           action="store_true", default=True)
    p.add_argument("--config",            type=str,   default=None,
                   help="Load configuration from YAML file (overrides CLI defaults)")
    return p


def _resolve(cli_val, cfg_val, default):
    """Return cli_val if explicitly set (not None), else cfg_val if present, else default."""
    if cli_val is not None:
        return cli_val
    if cfg_val is not None:
        return cfg_val
    return default


def _build_args(cli_args, config: dict):
    """Merge CLI arguments with loaded config, CLI taking precedence."""

    class Args:
        pass

    args = Args()
    cfg_prng = config.get("prng_comparison", {})
    cfg_api  = config.get("api", {})
    cfg_feat = config.get("features", {})
    cfg_vis  = config.get("visualization", {})
    cfg_aud  = config.get("audio", {})
    cfg_anom = config.get("anomaly", {})

    args.duration         = _resolve(cli_args.duration,        config.get("duration"),        None)
    args.output_dir       = _resolve(cli_args.output_dir,      config.get("output_dir"),      "./output")
    args.render_interval  = _resolve(cli_args.render_interval, cfg_vis.get("render_interval"), 60.0)
    args.prng             = cli_args.prng or cfg_prng.get("enabled", False)
    args.prng_gen         = _resolve(cli_args.prng_gen,        cfg_prng.get("generator"),     "pcg64")
    args.prng_seed        = _resolve(cli_args.prng_seed,       cfg_prng.get("seed"),          None)
    args.batch_size       = _resolve(cli_args.batch_size,      config.get("batch_size"),      1024)
    args.window           = _resolve(cli_args.window,          config.get("window"),          512)
    args.step             = _resolve(cli_args.step,            config.get("step"),            128)
    args.no_audio         = cli_args.no_audio  or not cfg_aud.get("enabled", True)
    args.no_anomaly       = cli_args.no_anomaly or not cfg_anom.get("enabled", True)
    args.anomaly_no_audio = cli_args.anomaly_no_audio
    args.verbose          = cli_args.verbose
    args.poll_interval    = cfg_api.get("poll_interval", 0.5)
    args.retry_max        = cfg_api.get("retry_max_attempts", 3)
    args.retry_backoff    = cfg_api.get("retry_backoff_base", 2.0)
    args.history_size     = cfg_feat.get("history", 200)

    return args


def print_banner(args) -> None:
    mode = f"QRNG + {args.prng_gen.upper()} PRNG (dual-source)" if args.prng else "QRNG only"
    print("=" * 65)
    print("  QRNG Sonifier - Research Demo")
    print(f"  Mode     : {mode}")
    print(f"  Audio    : {'disabled' if args.no_audio else 'enabled'}")
    print(f"  Anomaly  : {'disabled' if args.no_anomaly else 'enabled'}")
    print(f"  Output   : {args.output_dir}")
    print("=" * 65)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(name)s: %(message)s',
    )

    cli_args = build_parser().parse_args()

    config: dict = {}
    if cli_args.config:
        config = load_config(cli_args.config)
        logger.info("Loaded configuration from %s", cli_args.config)

    args = _build_args(cli_args, config)
    print_banner(args)

    # ── Sources ───────────────────────────────────────────────────────
    qrng_ingestor = QRNGIngestor(
        batch_size=args.batch_size,
        poll_interval=args.poll_interval,
        retry_max_attempts=args.retry_max,
        retry_backoff_base=args.retry_backoff,
        verbose=args.verbose,
    )
    prng_source = (
        PRNGSource(
            generator=args.prng_gen,
            batch_size=args.batch_size,
            poll_interval=0.5,
            seed=args.prng_seed,
            verbose=args.verbose,
        )
        if args.prng else None
    )

    # ── Feature engines ───────────────────────────────────────────────
    qrng_engine = FeatureEngine(window=args.window, step=args.step, history=args.history_size)
    prng_engine = FeatureEngine(window=args.window, step=args.step, history=args.history_size) if args.prng else None

    # ── Output layers ─────────────────────────────────────────────────
    renderer = Renderer(output_dir=args.output_dir, render_interval=args.render_interval)
    sonifier = Sonifier(dual_source=args.prng) if not args.no_audio else None
    detector = (
        AnomalyDetector(
            output_dir=args.output_dir,
            audio=(not args.anomaly_no_audio) and (not args.no_audio),
        )
        if not args.no_anomaly else None
    )

    # ── Graceful shutdown ──────────────────────────────────────────────
    shutdown = {"requested": False}

    def _handle(sig, frame):
        logger.info("Shutdown signal received...")
        shutdown["requested"] = True

    signal.signal(signal.SIGINT,  _handle)
    signal.signal(signal.SIGTERM, _handle)

    # ── Start ─────────────────────────────────────────────────────────
    qrng_ingestor.start()
    if prng_source:
        prng_source.start()
    if sonifier:
        sonifier.start()

    logger.info("Waiting for first batch...")

    start_time         = time.time()
    last_render        = start_time
    total_samples_qrng = 0
    total_samples_prng = 0
    total_frames       = 0
    anomaly_counts: dict = {}

    # ── Main loop ──────────────────────────────────────────────────────
    try:
        while not shutdown["requested"]:
            if args.duration and (time.time() - start_time) >= args.duration:
                logger.info("Duration %.1fs reached.", args.duration)
                break

            qrng_batch = qrng_ingestor.get_batch(timeout=10.0)
            if qrng_batch is None:
                logger.warning("Timeout on QRNG batch — retrying...")
                continue
            total_samples_qrng += len(qrng_batch)

            prng_batch = None
            if prng_source:
                prng_batch = prng_source.get_batch(timeout=2.0)
                if prng_batch is not None:
                    total_samples_prng += len(prng_batch)

            qrng_frames = qrng_engine.push(qrng_batch)
            prng_frames = (
                prng_engine.push(prng_batch)
                if (prng_engine and prng_batch is not None)
                else []
            )

            paired = (
                list(zip(qrng_frames, prng_frames))
                if prng_frames
                else [(f, None) for f in qrng_frames]
            )

            for qf, pf in paired:
                total_frames += 1

                kl = 0.0
                if pf is not None:
                    kl = AnomalyDetector._kl_divergence(qf.raw_window, pf.raw_window)

                renderer.push_frame(qf, qf.raw_window)

                if sonifier:
                    sonifier.push_frame(qf, prng_frame=pf, kl_divergence=kl)

                if detector:
                    events = detector.push(qf, prng_frame=pf)
                    for e in events:
                        anomaly_counts[e.trigger_name] = anomaly_counts.get(e.trigger_name, 0) + 1

                if args.verbose:
                    kl_str = f"  KL={kl:.4f}" if pf else ""
                    print(
                        f"[main] {total_frames:5d} | "
                        f"H={qf.entropy:.3f}  "
                        f"bias={qf.bias:+.3f}  "
                        f"AC={qf.autocorr_lag1:+.3f}  "
                        f"Hurst={qf.hurst:.3f}  "
                        f"runs_z={qf.runs_z:+.2f}"
                        f"{kl_str}"
                    )

            if time.time() - last_render >= args.render_interval:
                renderer.render_spectrogram(suffix=f"_frame{total_frames:05d}")
                last_render = time.time()

    finally:
        logger.info("Shutting down...")
        qrng_ingestor.stop()
        if prng_source:
            prng_source.stop()
        if sonifier:
            sonifier.stop()
        if total_frames >= 4:
            renderer.render_spectrogram(suffix="_final")
        renderer.close()
        if detector:
            detector.close()

        elapsed = time.time() - start_time
        print(f"\n{'=' * 65}")
        print(f"  Session summary")
        print(f"  Runtime          : {elapsed:.1f}s")
        print(f"  QRNG samples     : {total_samples_qrng:,}")
        if prng_source:
            print(f"  PRNG samples     : {total_samples_prng:,}")
        print(f"  Feature frames   : {total_frames:,}")
        if anomaly_counts:
            print(f"  Anomaly events   :")
            for name, cnt in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
                print(f"    {name:24s} x {cnt}")
        print(f"  Output dir       : {args.output_dir}")
        print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
