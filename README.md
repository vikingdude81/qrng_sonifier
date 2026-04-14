# QRNG Sonifier

Real-time audio synthesis and visualization tool for Quantum Random Number Generator (QRNG) data from ANU QRNG API. Features statistical analysis, anomaly detection, live spectrograms, and multi-modal interfaces.

## Features

- **Real-time Audio Synthesis**: Feature-driven FM synthesis with stereo spatialization
- **Statistical Analysis**: 9 different randomness metrics including entropy, Hurst exponent, chi-square, serial correlation, gap test
- **Anomaly Detection**: 10 trigger types with configurable thresholds and audio events
- **Dual-source Mode**: Compare QRNG vs PRNG with KL divergence monitoring
- **Publication-quality Output**: High-DPI spectrograms and rolling metrics CSV
- **Multi-interface Support**: CLI, GUI (tkinter), and Web dashboard (Flask)

## Installation

```bash
pip install -r requirements.txt
```

### Optional Dependencies

- `sounddevice`, `soundfile` for audio output
- `flask`, `flask-socketio` for web dashboard
- `pyyaml` for config file support

## Usage

### Headless Mode (CLI)

Generate metrics and spectrograms without audio:

```bash
python main.py --no-audio --duration 60 --render-interval 15
```

**Options:**
| Flag | Description | Default |
|------|-------------|---------|
| `--no-audio` | Disable audio output | False |
| `--duration N` | Run for N seconds (None=indefinite) | None |
| `--render-interval N` | Export spectrogram every N seconds | 60 |
| `--output-dir DIR` | Output directory for results | ./output |
| `--window SIZE` | Analysis window size | 512 |
| `--step SIZE` | Frame step interval | 128 |

### Dual-source Mode (QRNG vs PRNG)

```bash
python main.py --prng --prng-gen mt19937 --no-audio --duration 60
```

**PRNG Generators:** `mt19937`, `pcg64`, `sfc64`

### GUI Mode

Launch the desktop application:

```bash
python gui_app.py
```

Features real-time visualization with matplotlib embedded in Tkinter.

### Web Dashboard

Start the web server (requires Flask):

```bash
cd web_dashboard
python app.py
```

Then open http://127.0.0.1:5000 in your browser for live monitoring with Plotly charts.

### Using Configuration File

```bash
python main.py --config config.yaml
```

See `config.yaml` for all available settings including anomaly triggers, audio parameters, and API configuration.

## Statistical Features

| Feature | Range | Description |
|---------|-------|-------------|
| `entropy` | [0,1] | Shannon entropy normalized to max 8 bits |
| `bias` | [-1,1] | Mean deviation from uniform center (127.5) |
| `autocorr_lag1` | [-1,1] | Lag-1 autocorrelation |
| `runs_z` | [-5,5] | Runs test z-score for randomness |
| `hurst` | [0,1] | Hurst exponent via R/S analysis |
| `spectral_flat` | [0,1] | Wiener spectral flatness of histogram |
| `chi_square` | [0,1] | Chi-square goodness of fit to uniform |
| `serial_corr` | [-1,1] | Serial correlation at lag 5 |
| `gap_test_z` | [0,1] | Gap test z-score for randomness |

## Anomaly Triggers

| Trigger | Threshold | Severity | Audio Event |
|---------|-----------|----------|-------------|
| `entropy_drop` | < 0.85 | critical | low_tone |
| `entropy_surge` | > 0.999 | info | chime |
| `bias_drift` | > 0.15 | warning | sweep |
| `autocorr_spike` | > 0.12 | warning | burst |
| `runs_anomaly` | > 3.0 | warning | click |
| `hurst_persist` | > 0.65 | warning | sweep |
| `hurst_antipersist` | < 0.35 | info | sweep |
| `spectral_narrow` | < 0.50 | warning | burst |
| `source_diverge` | < 0.05 (KL) | critical | chime |
| `regime_change` | > 0.30 | warning | click |

## Output Files

Generated in `./output/`:

- `metrics_<timestamp>.csv` - Rolling feature values per frame
- `anomaly_log_<timestamp>.csv` - Detected anomalies with timestamps
- `spectrogram_<timestamp>.png` - Publication-quality dual-panel spectrogram (300 DPI)

## Testing

Run the test suite:

```bash
pytest tests/ -v
```

Tests cover all feature extraction algorithms with statistical validation.

## Architecture

```.
qrng_sonifier/
├── main.py              # Entry point, CLI handling
├── gui_app.py           # Tkinter desktop GUI
├── sonifier.py          # Audio synthesis engine
├── renderer.py          # Spectrogram & CSV export
├── features.py          # Statistical feature extraction
├── anomaly_triggers.py  # Anomaly detection logic
├── prng_source.py       # PRNG generators for comparison
├── ingest.py            # API polling with retry logic
├── config.yaml          # Configuration template
├── web_dashboard/       # Flask web app
│   ├── app.py           # Web server & WebSocket
│   └── templates/       # HTML templates
├── tests/               # Unit test suite
│   ├── __init__.py
│   └── test_features.py
└── output/              # Generated results
```

## API Endpoints (Web Dashboard)

| Endpoint | Description |
|----------|-------------|
| `/` | Main dashboard UI |
| `/api/frames` | JSON array of recent feature frames |
| `/api/latest` | Most recent frame data |
| `/api/anomalies` | Recent anomaly events |
| `/api/stats` | System statistics |

## License

MIT License

## Acknowledgments

- ANU Quantum Random Numbers Service (https://qrng.anu.edu.au/)
- Statistical tests based on NIST SP 800-22 randomness guidelines