"""
app.py — Web Dashboard for QRNG Sonifier with WebSocket Bridge integration.

Provides a real-time web interface with:
- Live feature plots using Plotly
- Spectrogram visualization  
- Anomaly event feed
- Configuration controls
- Real-time streaming from CLI via WebSocket bridge
"""

from __future__ import annotations
import json
import threading
import time
import os
from flask import Flask, render_template, jsonify, Response
from flask_socketio import SocketIO, emit
import numpy as np
import plotly.graph_objects as go
from plotly.utils import PlotlyJSONEncoder


class QRNGDashboard:
    """WebSocket-enabled web dashboard for real-time QRNG monitoring."""

    def __init__(self, host="127.0.0.1", port=5000):
        self.host = host
        self.port = port
        
        # Flask app
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'qrng-sonifier-secret'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*", async_mode='threading')
        
        # Data storage (thread-safe)
        self._lock = threading.Lock()
        self._frames: list = []
        self._anomaly_events: list = []
        self._latest_frame: dict = None
        
        # Streaming state
        self._streaming = False
        self._last_update = 0.0

    def setup_routes(self) -> None:
        """Set up Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/frames')
        def get_frames():
            with self._lock:
                frames = self._frames[-100:]  # Last 100 frames
            
            data = []
            for f in frames:
                data.append({
                    'index': f['index'],
                    'entropy': f['entropy'],
                    'bias': f['bias'],
                    'autocorr_lag1': f['autocorr_lag1'],
                    'runs_z': f['runs_z'],
                    'hurst': f['hurst'],
                    'spectral_flat': f['spectral_flat'],
                    'chi_square': f.get('chi_square', 0.0),
                    'serial_corr': f.get('serial_corr', 0.0),
                    'gap_test_z': f.get('gap_test_z', 0.0),
                })
            
            return jsonify({'frames': data, 'count': len(data)})
        
        @self.app.route('/api/latest')
        def get_latest():
            with self._lock:
                if self._latest_frame:
                    return jsonify(self._latest_frame)
                return jsonify({})
        
        @self.app.route('/api/anomalies')
        def get_anomalies():
            with self._lock:
                events = self._anomaly_events[-50:]
            
            return jsonify({'events': events, 'count': len(events)})
        
        @self.app.route('/api/stats')
        def get_stats():
            with self._lock:
                stats = {
                    'total_frames': len(self._frames),
                    'streaming': self._streaming,
                    'last_update': self._last_update,
                    'uptime': time.time() - getattr(self, '_start_time', 0) if (self._start_time := getattr(self, '_start_time', None)) else 0,
                }
            return jsonify(stats)

    def update_frame(self, frame_data: dict) -> None:
        """Update with a new feature frame."""
        with self._lock:
            self._frames.append(frame_data)
            self._latest_frame = {
                'index': frame_data['index'],
                'timestamp': time.time(),
                **{k: v for k, v in frame_data.items() if k != 'raw_window'}
            }

    def update_anomaly(self, event_data: dict) -> None:
        """Update with a new anomaly event."""
        with self._lock:
            self._anomaly_events.append({
                **event_data,
                'timestamp': time.time()
            })

    def start_streaming(self) -> None:
        """Enable real-time streaming."""
        with self._lock:
            self._streaming = True
            self._start_time = time.time()

    def stop_streaming(self) -> None:
        """Disable real-time streaming."""
        with self._lock:
            self._streaming = False

    # Socket event handlers for CLI integration
    @socketio.on('cli_frame_update')
    def handle_cli_frame(data):
        """Receive frame updates from CLI WebSocket bridge."""
        dashboard.update_frame(data)
        emit('frame_update', data, broadcast=True)

    @socketio.on('cli_anomaly_event')
    def handle_cli_anomaly(data):
        """Receive anomaly events from CLI WebSocket bridge."""
        dashboard.update_anomaly(data)
        emit('anomaly_event', data, broadcast=True)


# Global dashboard instance for CLI integration
dashboard = None


def get_dashboard() -> QRNGDashboard:
    """Get or create the global dashboard instance."""
    global dashboard
    if dashboard is None:
        dashboard = QRNGDashboard(host="127.0.0.1", port=5000)
    return dashboard


def generate_templates():
    """Generate the HTML templates for the dashboard."""
    
    os.makedirs('web_dashboard/templates', exist_ok=True)
    
    html_content = '''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>QRNG Sonifier Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.6.1/socket.io.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: Arial, sans-serif; background: #1a1a2e; color: #eee; }
        .header { background: #16213e; padding: 20px; text-align: center; border-bottom: 2px solid #0f3460; }
        .container { display: grid; grid-template-columns: 2fr 1fr; gap: 20px; padding: 20px; max-width: 1800px; margin: 0 auto; }
        .panel { background: #16213e; border-radius: 10px; padding: 20px; margin-bottom: 20px; }
        .panel h2 { color: #e94560; margin-bottom: 15px; font-size: 1.2em; }
        #main_plot { width: 100%; height: 400px; }
        .stat-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 15px; }
        .stat-card { background: #0f3460; padding: 15px; border-radius: 8px; text-align: center; }
        .stat-value { font-size: 2em; color: #e94560; font-weight: bold; }
        .stat-label { font-size: 0.9em; color: #aaa; margin-top: 5px; }
        .anomaly-list { max-height: 300px; overflow-y: auto; }
        .anomaly-item { background: #0f3460; padding: 10px; border-radius: 5px; margin-bottom: 8px; border-left: 4px solid #e94560; }
        .anomaly-item.info { border-left-color: #2ecc71; }
        .anomaly-item.warning { border-left-color: #f39c12; }
        .anomaly-item.critical { border-left-color: #e74c3c; }
        .anomaly-name { font-weight: bold; }
        .anomaly-severity { font-size: 0.8em; padding: 2px 6px; border-radius: 3px; display: inline-block; margin-top: 5px; }
        .severity-info { background: #2ecc71; color: #000; }
        .severity-warning { background: #f39c12; color: #000; }
        .severity-critical { background: #e74c3c; color: #fff; }
        .status-bar { display: flex; justify-content: space-between; align-items: center; padding: 15px 20px; background: #16213e; border-radius: 10px; margin-bottom: 20px; }
        .status-indicator { width: 12px; height: 12px; border-radius: 50%; display: inline-block; margin-right: 8px; }
        .status-active { background: #2ecc71; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.5; } }
    </style>
</head>
<body>
    <div class="header">
        <h1>QRNG Sonifier - Real-time Dashboard</h1>
    </div>
    
    <div class="status-bar">
        <span><span id="status-indicator" class="status-indicator"></span><span id="streaming-status">Connecting...</span></span>
        <span id="frame-count">Frames: 0</span>
        <span id="uptime">Uptime: 0s</span>
    </div>

    <div class="container">
        <div class="main-column">
            <div class="panel">
                <h2>Feature Timeline (Last 100 Frames)</h2>
                <div id="main_plot"></div>
            </div>
            
            <div class="panel">
                <h2>Additional Statistics</h2>
                <div class="stat-grid" style="margin-top: 15px;">
                    <div class="stat-card">
                        <div class="stat-value" id="chi-square">0.000</div>
                        <div class="stat-label">Chi-Square Deviation</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="serial-corr">0.000</div>
                        <div class="stat-label">Serial Correlation (lag 5)</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="gap-test">0.000</div>
                        <div class="stat-label">Gap Test Z-Score</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="side-column">
            <div class="panel">
                <h2>Anomaly Events</h2>
                <div id="anomaly-list" class="anomaly-list"></div>
            </div>
            
            <div class="panel">
                <h2>Live Statistics</h2>
                <div class="stat-grid">
                    <div class="stat-card">
                        <div class="stat-value" id="entropy-val">0.000</div>
                        <div class="stat-label">Entropy</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="bias-val">0.000</div>
                        <div class="stat-label">Bias</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-value" id="hurst-val">0.000</div>
                        <div class="stat-label">Hurst</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        const socket = io();
        
        function initPlot() {
            Plotly.newPlot('main_plot', [], {
                margin: { t: 30, r: 20, l: 50, b: 40 },
                showlegend: true,
                legend: { x: 0.98, y: 0.98 }
            });
        }

        function updatePlot(frames) {
            const traces = [
                {
                    x: frames.map(f => f.index),
                    y: frames.map(f => f.entropy),
                    name: 'Entropy',
                    line: { color: '#e67e22' }
                },
                {
                    x: frames.map(f => f.index),
                    y: frames.map(f => f.bias * 5),
                    name: 'Bias (×5)',
                    line: { color: '#e74c3c', dash: 'dash' }
                },
                {
                    x: frames.map(f => f.index),
                    y: frames.map(f => f.hurst * 10),
                    name: 'Hurst (×10)',
                    line: { color: '#2ecc71' }
                },
                {
                    x: frames.map(f => f.index),
                    y: frames.map(f => f.spectral_flat * 5),
                    name: 'Spectral Flat (×5)',
                    line: { color: '#3498db', dash: 'dot' }
                }
            ];

            Plotly.react('main_plot', traces, {
                margin: { t: 30, r: 20, l: 60, b: 50 },
                xaxis: { title: 'Frame Index' },
                yaxis: { title: 'Value (scaled)' }
            });
        }

        function updateStats(frame) {
            document.getElementById('entropy-val').textContent = frame.entropy.toFixed(3);
            document.getElementById('bias-val').textContent = frame.bias.toFixed(3);
            document.getElementById('hurst-val').textContent = frame.hurst.toFixed(3);
            document.getElementById('chi-square').textContent = (frame.chi_square || 0).toFixed(3);
            document.getElementById('serial-corr').textContent = (frame.serial_corr || 0).toFixed(3);
            document.getElementById('gap-test').textContent = (frame.gap_test_z || 0).toFixed(3);
        }

        function updateAnomalies(events) {
            const container = document.getElementById('anomaly-list');
            
            if (!events.length) {
                container.innerHTML = '<p style="color: #666; text-align: center;">No anomalies</p>';
                return;
            }

            container.innerHTML = events.slice(-20).reverse().map(e => `
                <div class="anomaly-item ${e.severity}">
                    <span class="anomaly-name">${e.trigger}</span>
                    <span class="anomaly-severity severity-${e.severity}">${e.severity.toUpperCase()}</span>
                    <br><small>${new Date(e.timestamp * 1000).toLocaleTimeString()}</small>
                </div>
            `).join('');
        }

        socket.on('connect', () => {
            document.getElementById('streaming-status').textContent = 'Connected';
            document.getElementById('status-indicator').classList.add('status-active');
        });

        socket.on('frame_update', (data) => {
            updateStats(data);
            
            fetch('/api/frames')
                .then(r => r.json())
                .then(d => updatePlot(d.frames));
                
            document.getElementById('frame-count').textContent = `Frames: ${d?.count || data.index}`;
        });

        socket.on('anomaly_event', (data) => {
            fetch('/api/anomalies')
                .then(r => r.json())
                .then(d => updateAnomalies(d.events));
        });

        initPlot();
        
        setInterval(() => {
            const uptimeEl = document.getElementById('uptime');
            let seconds = 0;
            const counter = setInterval(() => {
                seconds++;
                uptimeEl.textContent = `Uptime: ${seconds}s`;
            }, 1000);
            
            socket.on('disconnect', () => clearInterval(counter));
        }, 1000);

        fetch('/api/frames')
            .then(r => r.json())
            .then(d => updatePlot(d.frames));
        
        fetch('/api/anomalies')
            .then(r => r.json())
            .then(d => updateAnomalies(d.events));
    </script>
</body>
</html>'''

    with open('web_dashboard/templates/index.html', 'w') as f:
        f.write(html_content)


def run_server(host="127.0.0.1", port=5000, debug=False):
    """Run the web dashboard server."""
    global dashboard
    
    # Create and setup dashboard
    dashboard = QRNGDashboard(host=host, port=port)
    dashboard.setup_routes()
    
    print(f"Starting QRNG Dashboard at http://{host}:{port}")
    socketio.run(dashboard.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


if __name__ == '__main__':
    generate_templates()
    run_server(debug=True)