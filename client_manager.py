# client_manager.py
"""
Client Session Manager - Optimized
Handles audio buffering, VAD chunking, and result delivery
"""
import asyncio
import numpy as np
import torch
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional


@dataclass
class Transcript:
    sequence: int
    text: str
    timestamp: float
    duration: float


class ClientSession:
    """Manages a single client's transcription session"""
    
    def __init__(self, session_id: str, sample_rate: int = 16000):
        self.session_id = session_id
        self.sample_rate = sample_rate
        
        # Audio buffer
        self.buffer = bytearray()
        self.buffer_lock = asyncio.Lock()
        
        # Transcripts
        self.transcripts: deque[Transcript] = deque(maxlen=1000)
        self.transcript_lock = asyncio.Lock()
        self.next_seq = 0
        
        # Processing state
        self.chunks_submitted = 0
        self.chunks_completed = 0
        self.state_lock = asyncio.Lock()
        
        # Metadata
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.total_bytes = 0
    
    async def add_audio(self, audio_bytes: bytes):
        """Add audio to buffer"""
        async with self.buffer_lock:
            self.buffer.extend(audio_bytes)
            self.last_activity = datetime.now()
            self.total_bytes += len(audio_bytes)
    
    def get_buffer(self) -> bytes:
        """Get buffer copy"""
        return bytes(self.buffer)
    
    async def clear_buffer(self, up_to: Optional[int] = None):
        """Clear buffer (partially or fully)"""
        async with self.buffer_lock:
            if up_to is None:
                self.buffer.clear()
            else:
                self.buffer = bytearray(self.buffer[up_to:])
    
    async def add_transcript(self, text: str, duration: float):
        """Add transcript result"""
        async with self.transcript_lock:
            t = Transcript(
                sequence=self.next_seq,
                text=text,
                timestamp=datetime.now().timestamp(),
                duration=duration
            )
            self.transcripts.append(t)
            self.next_seq += 1
    
    async def get_transcripts(self, since_seq: int, limit: int) -> List[dict]:
        """Get transcripts since sequence"""
        async with self.transcript_lock:
            results = [
                {
                    "sequence": t.sequence,
                    "text": t.text,
                    "timestamp": t.timestamp,
                    "duration": t.duration
                }
                for t in self.transcripts
                if t.sequence >= since_seq
            ]
            return results[:limit]
    
    def buffer_duration(self) -> float:
        """Get buffer duration in seconds"""
        return len(self.buffer) / (self.sample_rate * 2)
    
    def is_stale(self, timeout_min: int = 30) -> bool:
        """Check if session is stale"""
        return datetime.now() - self.last_activity > timedelta(minutes=timeout_min)
    
    async def inc_submitted(self):
        async with self.state_lock:
            self.chunks_submitted += 1
    
    async def inc_completed(self):
        async with self.state_lock:
            self.chunks_completed += 1
    
    async def get_state(self) -> dict:
        """Get processing state"""
        async with self.state_lock:
            return {
                "submitted": self.chunks_submitted,
                "completed": self.chunks_completed,
                "pending": self.chunks_submitted - self.chunks_completed,
                "is_processing": self.chunks_submitted > self.chunks_completed
            }


class VADChunker:
    """Voice Activity Detection based audio chunking"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        
        try:
            torch.set_num_threads(1)
            self.vad_model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            print("VAD model loaded")
        except Exception as e:
            print(f"VAD load failed: {e}")
            self.vad_model = None
            self.get_speech_timestamps = None
        
        # VAD parameters
        self.min_speech_ms = 300
        self.min_silence_ms = 500
        self.speech_pad_ms = 300
        
        # Chunk constraints
        self.min_chunk_sec = 1.0
        self.max_chunk_sec = 45.0
        self.min_rms = 0.005  # Silence threshold
    
    def find_chunks(self, audio_bytes: bytes) -> List[tuple]:
        """Find speech chunks in audio buffer"""
        
        # Ensure even length
        if len(audio_bytes) % 2:
            audio_bytes = audio_bytes[:-1]
        
        # Need minimum audio
        if len(audio_bytes) < self.sample_rate * 2:
            return []
        
        # Fallback: simple chunking if VAD unavailable
        if not self.vad_model:
            return self._simple_chunk(audio_bytes)
        
        try:
            # Convert to float32
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            
            # Get speech timestamps
            timestamps = self.get_speech_timestamps(
                torch.from_numpy(audio),
                self.vad_model,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=self.min_speech_ms,
                min_silence_duration_ms=self.min_silence_ms,
                speech_pad_ms=self.speech_pad_ms,
                return_seconds=False
            )
            
            if not timestamps:
                return []
            
            # Convert to byte positions
            chunks = []
            for ts in timestamps:
                start_byte = ts['start'] * 2
                end_byte = ts['end'] * 2
                dur_sec = (end_byte - start_byte) / (self.sample_rate * 2)
                
                # Filter by duration
                if self.min_chunk_sec <= dur_sec <= self.max_chunk_sec:
                    chunks.append((start_byte, end_byte))
            
            # Merge nearby chunks
            if len(chunks) > 1:
                merged = []
                current_start, current_end = chunks[0]
                
                for start, end in chunks[1:]:
                    gap_sec = (start - current_end) / (self.sample_rate * 2)
                    merged_dur = (end - current_start) / (self.sample_rate * 2)
                    
                    # Merge if gap < 1s and total < max
                    if gap_sec < 1.0 and merged_dur <= self.max_chunk_sec:
                        current_end = end
                    else:
                        merged.append((current_start, current_end))
                        current_start, current_end = start, end
                
                merged.append((current_start, current_end))
                chunks = merged
            
            # Filter silence
            filtered = []
            for start, end in chunks:
                chunk_bytes = audio_bytes[start:end]
                if self._check_rms(chunk_bytes):
                    filtered.append((start, end))
            
            return filtered
            
        except Exception as e:
            print(f"VAD error: {e}")
            return self._simple_chunk(audio_bytes)
    
    def _simple_chunk(self, audio_bytes: bytes) -> List[tuple]:
        """Simple duration-based chunking"""
        dur = len(audio_bytes) / (self.sample_rate * 2)
        
        if self.min_chunk_sec <= dur <= self.max_chunk_sec:
            if self._check_rms(audio_bytes):
                return [(0, len(audio_bytes))]
        
        return []
    
    def _check_rms(self, audio_bytes: bytes) -> bool:
        """Check if audio has sufficient energy"""
        if len(audio_bytes) < 2:
            return False
        
        try:
            audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio ** 2))
            return rms >= self.min_rms
        except:
            return False


class ClientManager:
    """Manages all client sessions"""
    
    def __init__(self, engine):
        self.engine = engine
        self.sessions: Dict[str, ClientSession] = {}
        self.session_locks: Dict[str, asyncio.Lock] = {}
        self.vad = VADChunker()
        
        # Transcription queue
        self.queue = asyncio.Queue(maxsize=100)
        self.worker_task = None
        
        # Stats
        self.total_sessions = 0
    
    async def create_session(self, session_id: str):
        """Create new session"""
        self.sessions[session_id] = ClientSession(session_id)
        self.session_locks[session_id] = asyncio.Lock()
        self.total_sessions += 1
        
        # Start worker if needed
        if not self.worker_task or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())
        
        print(f"Session created: {session_id}")
    
    async def session_exists(self, session_id: str) -> bool:
        """Check if session exists"""
        return session_id in self.sessions
    
    async def add_audio(self, session_id: str, audio_bytes: bytes, sequence: int):
        """Add audio to session buffer"""
        if session_id not in self.sessions:
            return
        
        session = self.sessions[session_id]
        await session.add_audio(audio_bytes)
        
        # Auto-process if buffer is large
        if session.buffer_duration() >= 5.0:
            await self._process_buffer(session_id)
    
    async def _process_buffer(self, session_id: str):
        """Process session buffer for speech chunks"""
        async with self.session_locks[session_id]:
            session = self.sessions[session_id]
            buffer = session.get_buffer()
            
            # Need minimum audio
            if len(buffer) < self.vad.sample_rate * 2:
                return
            
            # Find speech chunks
            chunks = self.vad.find_chunks(buffer)
            
            if not chunks:
                # Clear old silence if buffer too large
                if session.buffer_duration() > 10.0:
                    await session.clear_buffer(len(buffer) // 2)
                return
            
            # Queue chunks for transcription
            last_end = 0
            for start, end in chunks:
                chunk_bytes = buffer[start:end]
                
                await session.inc_submitted()
                
                try:
                    # Add to queue with timeout
                    await asyncio.wait_for(
                        self.queue.put((session_id, chunk_bytes)),
                        timeout=0.5
                    )
                    last_end = end
                    
                except asyncio.TimeoutError:
                    # Queue full, rollback counter
                    async with session.state_lock:
                        session.chunks_submitted -= 1
                    break
            
            # Clear processed audio
            if last_end > 0:
                await session.clear_buffer(last_end)
    
    async def flush_session(self, session_id: str):
        """Force buffer processing"""
        if session_id in self.sessions:
            await self._process_buffer(session_id)
    
    async def _worker(self):
        """Transcription worker"""
        print("Transcription worker started")
        
        while True:
            try:
                session_id, audio_bytes = await self.queue.get()
                
                # Validate
                if len(audio_bytes) % 2:
                    audio_bytes = audio_bytes[:-1]
                
                if not audio_bytes or session_id not in self.sessions:
                    self.queue.task_done()
                    if session_id in self.sessions:
                        await self.sessions[session_id].inc_completed()
                    continue
                
                # Convert to float32
                try:
                    audio = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                except ValueError as e:
                    print(f"Buffer conversion error [{session_id}]: {e}")
                    self.queue.task_done()
                    await self.sessions[session_id].inc_completed()
                    continue
                
                # Transcribe
                t0 = datetime.now()
                text = await self.engine.transcribe(audio)
                dur = (datetime.now() - t0).total_seconds()
                
                # Store result
                if text and session_id in self.sessions:
                    await self.sessions[session_id].add_transcript(text, dur)
                    print(f"[{session_id}] {text} ({dur:.2f}s)")
                
                # Update counter
                if session_id in self.sessions:
                    await self.sessions[session_id].inc_completed()
                
                self.queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Worker error: {e}")
                import traceback
                traceback.print_exc()
                
                if 'session_id' in locals() and session_id in self.sessions:
                    await self.sessions[session_id].inc_completed()
    
    async def get_transcripts(self, session_id: str, since_seq: int, limit: int) -> List[dict]:
        """Get transcripts for session"""
        if session_id not in self.sessions:
            return []
        return await self.sessions[session_id].get_transcripts(since_seq, limit)
    
    async def get_processing_state(self, session_id: str) -> dict:
        """Get processing state"""
        if session_id not in self.sessions:
            return {"error": "not found"}
        return await self.sessions[session_id].get_state()
    
    async def close_session(self, session_id: str):
        """Close and cleanup session"""
        if session_id not in self.sessions:
            return
        
        await self.flush_session(session_id)
        
        del self.sessions[session_id]
        del self.session_locks[session_id]
        
        print(f"Session closed: {session_id}")
    
    async def cleanup_stale_sessions(self):
        """Periodic cleanup of stale sessions"""
        while True:
            try:
                await asyncio.sleep(300)  # Every 5 min
                
                stale = [
                    sid for sid, session in self.sessions.items()
                    if session.is_stale(30)
                ]
                
                for sid in stale:
                    try:
                        await self.close_session(sid)
                        print(f"Cleaned stale session: {sid}")
                    except Exception as e:
                        print(f"Cleanup error [{sid}]: {e}")
                        
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"Cleanup task error: {e}")
    
    async def cleanup_all(self):
        """Cleanup all sessions"""
        for sid in list(self.sessions.keys()):
            await self.close_session(sid)
    
    def get_stats(self) -> dict:
        """Get manager stats"""
        return {
            "active_sessions": len(self.sessions),
            "queue_size": self.queue.qsize(),
            "total_sessions": self.total_sessions
        }