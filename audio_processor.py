"""
Audio Preprocessing Pipeline
- Noise gating
- Dynamic range compression
- Speech frequency band filtering
- Normalization
"""
import numpy as np
from scipy import signal
from typing import Optional


class AudioPreprocessor:
    """
    Preprocess incoming audio chunks for optimal transcription
    All operations are stateless per-chunk
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        # Noise gate parameters
        self.gate_threshold = 0.01  # -40dB
        self.gate_attack_samples = int(0.005 * sample_rate)  # 5ms
        self.gate_release_samples = int(0.05 * sample_rate)  # 50ms
        
        # Compression parameters
        self.comp_threshold = 0.3
        self.comp_ratio = 3.0
        self.comp_knee_width = 0.1
        
        # Speech band filter (80Hz - 8kHz)
        self.filter_low = 80
        self.filter_high = 8000
        self._design_filter()
        
    def _design_filter(self):
        """Design bandpass filter for speech frequencies"""
        nyquist = self.sample_rate / 2
        low = self.filter_low / nyquist
        high = self.filter_high / nyquist -0.01
        print(low, high)
        self.sos = signal.butter(
            4,  # 4th order
            [low, high],
            btype='bandpass',
            output='sos'
        )
    
    def process_chunk(self, audio_bytes: bytes) -> bytes:
        """
        Process raw audio chunk through full pipeline
        Input/Output: 16-bit PCM bytes
        """
        # Ensure buffer is aligned to int16 (2 bytes)
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]  # Drop last byte if odd
        
        if len(audio_bytes) == 0:
            return audio_bytes
            
        # Convert to float32 array
        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(audio) == 0:
            return audio_bytes
        
        # 1. Bandpass filter (speech frequencies)
        audio = self._apply_bandpass(audio)
        
        # 2. Noise gate
        audio = self._apply_gate(audio)
        
        # 3. Dynamic range compression
        audio = self._apply_compression(audio)
        
        # 4. Normalize
        audio = self._normalize(audio)
        
        # Convert back to int16
        audio_int16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        return audio_int16.tobytes()
    
    def _apply_bandpass(self, audio: np.ndarray) -> np.ndarray:
        """Apply speech frequency bandpass filter"""
        if len(audio) < 20:  # Too short to filter
            return audio
        
        try:
            filtered = signal.sosfilt(self.sos, audio)
            return filtered
        except Exception:
            return audio
    
    def _apply_gate(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply noise gate to reduce background noise
        Simple threshold-based gating with envelope following
        """
        # Calculate envelope
        envelope = np.abs(audio)
        
        # Smooth envelope
        if len(envelope) > self.gate_attack_samples:
            window = np.ones(self.gate_attack_samples) / self.gate_attack_samples
            envelope = np.convolve(envelope, window, mode='same')
        
        # Create gate mask
        gate_open = envelope > self.gate_threshold
        
        # Apply simple gate
        gated = audio * gate_open
        
        return gated
    
    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """
        Apply dynamic range compression
        Reduces loud sounds, preserves quiet ones
        """
        # Calculate magnitude
        magnitude = np.abs(audio)
        
        # Soft knee compression
        compressed_magnitude = np.where(
            magnitude > self.comp_threshold,
            self.comp_threshold + (magnitude - self.comp_threshold) / self.comp_ratio,
            magnitude
        )
        
        # Preserve sign
        compressed = np.sign(audio) * compressed_magnitude
        
        return compressed
    
    def _normalize(self, audio: np.ndarray, target_level: float = 0.7) -> np.ndarray:
        """
        Normalize audio to target level
        Prevents clipping while maximizing signal
        """
        max_val = np.abs(audio).max()
        
        if max_val > 1e-6:  # Avoid division by zero
            scale = target_level / max_val
            # Limit gain to prevent over-amplification of noise
            scale = min(scale, 10.0)
            audio = audio * scale
        
        return audio


class SimplePreprocessor:
    """
    Lightweight preprocessor for low-resource scenarios
    Just normalization and basic filtering
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
    
    def process_chunk(self, audio_bytes: bytes) -> bytes:
        """Minimal processing"""
        # Ensure buffer is aligned to int16 (2 bytes)
        if len(audio_bytes) % 2 != 0:
            audio_bytes = audio_bytes[:-1]
        
        if len(audio_bytes) == 0:
            return audio_bytes

        audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        
        if len(audio) == 0:
            return audio_bytes
        
        # Just normalize
        max_val = np.abs(audio).max()
        if max_val > 1e-6:
            audio = audio * (0.7 / max_val)
        
        audio_int16 = (audio * 32767.0).clip(-32768, 32767).astype(np.int16)
        return audio_int16.tobytes()