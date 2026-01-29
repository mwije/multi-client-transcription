"""
Transcription Engine
Thread-safe wrapper around Whisper model
"""
import asyncio
import numpy as np
import threading
from typing import Optional
from faster_whisper import WhisperModel


class TranscriptionEngine:
    def __init__(
        self,
        model_name: str = "small.en",
        device: str = "cpu",
        compute_type: str = "int8",
        cpu_threads: int = 6,
        num_workers: int = 1
    ):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.cpu_threads = cpu_threads
        self.model: Optional[WhisperModel] = None
        self.lock = threading.Lock()
        self._ready = False
    
    async def initialize(self):
        print(f"Loading Whisper model: {self.model_name}...")
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_model)
        self._ready = True
        print("Model loaded")
    
    def _load_model(self):
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=self.cpu_threads,
            num_workers=1
        )
    
    def is_ready(self) -> bool:
        return self._ready
    
    async def transcribe(self, audio: np.ndarray, language: str = "en") -> str:
        if not self._ready:
            raise RuntimeError("Model not initialized")
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._transcribe_sync, audio, language)
    
    def _transcribe_sync(self, audio: np.ndarray, language: str) -> str:
        with self.lock:
            segments, _ = self.model.transcribe(
                audio,
                language=language,
                beam_size=1,
                vad_filter=False,
                condition_on_previous_text=False
            )
            return " ".join(s.text for s in segments).strip()
    
    async def shutdown(self):
        self._ready = False
        print("Engine shutdown")