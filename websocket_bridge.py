"""
websocket_bridge.py — WebSocket bridge between CLI and web dashboard.

Enables real-time streaming of feature frames and anomaly events from the
CLI process to the Flask web dashboard via WebSocket connections.
"""

from __future__ import annotations
import threading
import time
import json
from typing import Callable, Optional, Deque, Dict, Any
from collections import deque
import numpy as np


class WebSocketBridge:
    """
    Bridges CLI data streams to WebSocket clients.
    
    This class acts as a central hub that:
    - Collects feature frames from the main process
    - Collects anomaly events from the detector
    - Broadcasts updates to all connected WebSocket clients
    
    Thread-safe for use alongside the QRNG sonifier's background threads.
    """

    def __init__(self, max_history: int = 500):
        self.max_history = max_history
        
        # Connected clients (thread-safe)
        self._clients: list = []
        self._lock = threading.Lock()
        
        # Data buffers
        self._frame_history: Deque[dict] = deque(maxlen=max_history)
        self._anomaly_events: Deque[dict] = deque(maxlen=200)
        
        # Callbacks for external data injection
        self._on_frame_callback: Optional[Callable[[np.ndarray, dict], None]] = None
        
    def register_client(self, client_id: str) -> None:
        """Register a new WebSocket client."""
        with self._lock:
            if client_id not in self._clients:
                self._clients.append(client_id)
    
    def unregister_client(self, client_id: str) -> None:
        """Unregister a WebSocket client."""
        with self._lock:
            if client_id in self._clients:
                self._clients.remove(client_id)
    
    def get_client_count(self) -> int:
        """Get number of connected clients."""
        with self._lock:
            return len(self._clients)
    
    def push_frame(self, frame_data: dict) -> None:
        """
        Push a new feature frame to the bridge.
        
        Parameters
        ----------
        frame_data : dict
            Dictionary containing frame data including 'index' and all features.
        """
        with self._lock:
            # Store frame without raw_window for memory efficiency
            clean_frame = {k: v for k, v in frame_data.items() if k != 'raw_window'}
            self._frame_history.append(clean_frame)
            
            # Notify all clients
            for client_id in list(self._clients):
                try:
                    self._emit_to_client(client_id, 'frame_update', clean_frame)
                except Exception as e:
                    print(f"[WebSocketBridge] Error emitting to {client_id}: {e}")
    
    def push_anomaly(self, event_data: dict) -> None:
        """
        Push a new anomaly event to the bridge.
        
        Parameters
        ----------
        event_data : dict
            Dictionary containing 'trigger', 'severity', and other details.
        """
        with self._lock:
            self._anomaly_events.append({
                **event_data,
                'timestamp': time.time()
            })
            
            # Notify all clients
            for client_id in list(self._clients):
                try:
                    self._emit_to_client(client_id, 'anomaly_event', event_data)
                except Exception as e:
                    print(f"[WebSocketBridge] Error emitting to {client_id}: {e}")
    
    def _emit_to_client(self, client_id: str, event_type: str, data: dict) -> None:
        """Emit an event to a specific client (must hold lock)."""
        # This would integrate with Flask-SocketIO in production
        # For now, we store the callback for testing
        if self._on_frame_callback and event_type == 'frame_update':
            self._on_frame_callback(client_id, data)
    
    def get_frames(self, limit: int = 100) -> list:
        """Get recent frames."""
        with self._lock:
            return list(self._frame_history)[-limit:]
    
    def get_anomalies(self, limit: int = 50) -> list:
        """Get recent anomaly events."""
        with self._lock:
            return list(self._anomaly_events)[-limit:]
    
    def get_stats(self) -> dict:
        """Get bridge statistics."""
        with self._lock:
            return {
                'connected_clients': len(self._clients),
                'frames_in_history': len(self._frame_history),
                'anomalies_in_history': len(self._anomaly_events),
                'uptime': time.time() - getattr(self, '_start_time', time.time())
            }

    def set_callback(self, callback: Callable[[str, dict], None]) -> None:
        """Set a callback for emitting events (for testing)."""
        self._on_frame_callback = callback
    
    def start(self) -> None:
        """Start the bridge."""
        self._start_time = time.time()


class SocketIOServerBridge:
    """
    Flask-SocketIO integration for QRNG Sonifier.
    
    This class provides a drop-in replacement that integrates with
    Flask-SocketIO to enable real-time WebSocket communication.
    """

    def __init__(self, socketio, host="127.0.0.1", port=8765):
        self.socketio = socketio
        self.host = host
        self.port = port
        self._bridge = WebSocketBridge()
        
        # Register event handlers
        @socketio.on('connect')
        def on_connect():
            client_id = request.sid if 'request' in dir() else "unknown"
            self._bridge.register_client(client_id)
            print(f"[WebSocket] Client connected: {client_id}")
            
            # Send current stats
            self.socketio.emit('stats', self._bridge.get_stats())
        
        @socketio.on('disconnect')
        def on_disconnect():
            client_id = request.sid if 'request' in dir() else "unknown"
            self._bridge.unregister_client(client_id)
            print(f"[WebSocket] Client disconnected: {client_id}")

    def push_frame(self, frame_data: dict) -> None:
        """Push a feature frame to all clients."""
        self.socketio.emit('frame_update', frame_data)
    
    def push_anomaly(self, event_data: dict) -> None:
        """Push an anomaly event to all clients."""
        self.socketio.emit('anomaly_event', event_data)

    def get_bridge(self) -> WebSocketBridge:
        """Get the underlying bridge instance."""
        return self._bridge


# Example usage and integration
def create_cli_with_websocket():
    """Create a CLI app that streams to WebSocket clients."""
    
    from flask import request
    
    socketio = SocketIO(cors_allowed_origins="*")
    
    @socketio.on('get_frames')
    def handle_get_frames(data=None):
        return jsonify(bridge.get_frames())
    
    @socketio.on('get_anomalies')
    def handle_get_anomalies(data=None):
        return jsonify(bridge.get_anomalies())
    
    bridge = SocketIOServerBridge(socketio)
    
    # In the main loop, call:
    # bridge.push_frame(frame_data)
    # bridge.push_anomaly(event_data)
    
    return socketio, bridge


if __name__ == '__main__':
    # Test script for WebSocket bridge
    print("WebSocket Bridge Test")
    print("=" * 50)
    
    bridge = WebSocketBridge()
    bridge.start()
    
    # Simulate frame data
    for i in range(10):
        frame_data = {
            'index': i,
            'entropy': np.random.uniform(0.8, 1.0),
            'bias': np.random.uniform(-0.2, 0.2),
            'autocorr_lag1': np.random.uniform(-0.1, 0.1),
            'runs_z': np.random.normal(0, 1),
            'hurst': np.random.uniform(0.4, 0.6),
            'spectral_flat': np.random.uniform(0.3, 0.7),
        }
        
        bridge.push_frame(frame_data)
        print(f"Pushed frame {i}: entropy={frame_data['entropy']:.3f}")
    
    # Simulate anomaly event
    anomaly_data = {
        'trigger': 'entropy_drop',
        'severity': 'warning',
        'value': 0.75,
        'frame_index': 8
    }
    bridge.push_anomaly(anomaly_data)
    print(f"\nPushed anomaly: {anomaly_data['trigger']}")
    
    # Print stats
    print("\nBridge Stats:")
    for k, v in bridge.get_stats().items():
        print(f"  {k}: {v}")
    
    # Test history retrieval
    print("\nRecent Frames (last 5):")
    for frame in bridge.get_frames(5):
        print(f"  Frame {frame['index']}: entropy={frame['entropy']:.3f}")