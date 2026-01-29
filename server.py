# server.py
"""
REST API Server for Multi-Client Real-Time Transcription
Optimized for reliability and performance
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import Optional

from fastapi import FastAPI, HTTPException, UploadFile, File, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from engine import TranscriptionEngine
from client_manager import ClientManager
from audio_processor import AudioPreprocessor

app = FastAPI(title="Transcription Service")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances
engine = TranscriptionEngine()
client_manager = ClientManager(engine)
preprocessor = AudioPreprocessor()


class SessionCreateResponse(BaseModel):
    session_id: str
    expires_at: str


class TranscriptResponse(BaseModel):
    transcripts: list
    has_more: bool


@app.on_event("startup")
async def startup():
    """Initialize transcription engine"""
    await engine.initialize()
    asyncio.create_task(client_manager.cleanup_stale_sessions())


@app.on_event("shutdown")
async def shutdown():
    """Cleanup resources"""
    await engine.shutdown()
    await client_manager.cleanup_all()


@app.post("/session/create", response_model=SessionCreateResponse)
async def create_session():
    """Create new transcription session"""
    session_id = str(uuid.uuid4())
    await client_manager.create_session(session_id)
    
    expires_at = datetime.now() + timedelta(hours=2)
    
    return SessionCreateResponse(
        session_id=session_id,
        expires_at=expires_at.isoformat()
    )


@app.post("/session/{session_id}/audio")
async def upload_audio_chunk(
    session_id: str,
    file: UploadFile = File(...),
    sequence: Optional[int] = Header(None, alias="X-Sequence-Number")
):
    """
    Upload audio chunk (16kHz mono 16-bit PCM)
    Sequence number helps with ordering but is optional
    """
    if not await client_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    try:
        audio_data = await file.read()
        
        if len(audio_data) == 0:
            raise HTTPException(status_code=400, detail="Empty audio")
        
        # Preprocess and add to queue
        processed = preprocessor.process_chunk(audio_data)
        await client_manager.add_audio(session_id, processed, sequence or 0)
        
        return {"status": "accepted", "bytes": len(audio_data)}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Audio processing error [{session_id}]: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}/transcripts", response_model=TranscriptResponse)
async def get_transcripts(
    session_id: str,
    since_sequence: int = 0,
    limit: int = 100
):
    """Get transcripts since sequence number"""
    if not await client_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    transcripts = await client_manager.get_transcripts(session_id, since_sequence, limit)
    
    return TranscriptResponse(
        transcripts=transcripts,
        has_more=len(transcripts) >= limit
    )


@app.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get processing status"""
    if not await client_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    state = await client_manager.get_processing_state(session_id)
    
    return {
        "session_id": session_id,
        "chunks_submitted": state["submitted"],
        "chunks_completed": state["completed"],
        "chunks_pending": state["pending"],
        "is_processing": state["is_processing"]
    }


@app.post("/session/{session_id}/flush")
async def flush_session(session_id: str):
    """Force processing of buffered audio"""
    if not await client_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    await client_manager.flush_session(session_id)
    
    return {"status": "flushed"}


@app.delete("/session/{session_id}")
async def close_session(session_id: str):
    """Close session and cleanup"""
    if not await client_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    
    await client_manager.close_session(session_id)
    
    return {"status": "closed"}


@app.get("/health")
async def health_check():
    """Service health check"""
    stats = client_manager.get_stats()
    return {
        "status": "healthy",
        "active_sessions": stats["active_sessions"],
        "queue_size": stats["queue_size"],
        "model_ready": engine.is_ready()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, workers=1)