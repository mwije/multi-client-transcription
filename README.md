# 🎙️ Local Multi-Client Transcription Service

A **local, session-based transcription service** designed to support **multiple concurrent clients** using a shared speech-to-text engine, with particular attention to **privacy, constrained compute environments, and clinical dictation workflows**.

>This repository represents an **exploratory implementation**, developed for evaluating on-premise transcription for healthcare use cases (e.g. radiology reporting) in a resource constrained setting.

## Overview

Many healthcare institutions face dilemmas opting in for clinical dictation in healthcare workflows (e.g. radiology, clinic notes)
* Cloud-based transcription raises privacy, governance, and cost concerns
* On-premise IT infrastructure are often **CPU-constrained**
* Clinicians expect **incremental feedback**, not long batch delays

This project explores whether a **locally hosted transcription service**, shared across users, can provide **acceptable latency and reliability** under these constraints.

This is a **technical exploration**, not a production system.


## Why This Repository Exists

This project reflects a practical attempt to reason through privacy, performance, and usability under real constraints. It is a stepping stone to build shared intuition and understanding before committing to heavier architectures or external services. In resource constrained healthcare settings, such exploration is a necessary first step.

This repository shares:

* a **technical exploration**
* a **learning artefact** and
* a **reference implementation** for local transcription experiments

---
## Design Goals

* **Local-first**: all audio processing occurs on the local machine or network
* **Multi-client**: multiple users can connect concurrently via isolated sessions
* **CPU-first**: designed and tested under CPU-constrained conditions
* **Incremental transcription**: partial results delivered during dictation
* **Operational clarity**: explicit session lifecycle and observable state

Non-priorities: high availability, persistence, security hardening.

## High Level Architecture

```
Browser Clients
 (Mic / Audio File)
        │
        │ HTTP (local network)
        ▼
┌───────────────────────────┐
│        FastAPI Server     │
│───────────────────────────│
│ • Session management      │
│ • Per-session queues      │
│ • Status & transcript API │
│ • Shared engine access    │
└───────────▲───────────────┘
            │
┌───────────┴───────────────┐
│  Transcription Engine     │
│───────────────────────────│
│ • CPU-first inference     │
│ • Shared across sessions  │
│ • Model-configurable      │
└───────────────────────────┘
```

A single transcription engine instance is shared across client sessions to reflect realistic on-premise hardware limits.


## Dictation & Input Modes

The browser client supports two primary interaction patterns:

### 🎤 Live Dictation

* **Sentence mode**
  Push-to-talk interaction where audio is sent after each utterance.
  Suitable for structured reporting and careful phrasing.

* **Continuous mode**
  Audio is streamed in fixed-duration chunks during active recording.
  Better suited for free-flow dictation.

### 📁 Audio File Upload

* Client-side decoding and resampling
* Chunked upload with progress feedback
* Incremental transcription as processing progresses

All audio is normalized to **16 kHz mono PCM** before server-side processing.


## API & Interaction Model

* Clients explicitly create and close sessions
* Audio is uploaded in discrete chunks
* Transcripts are retrieved incrementally via polling
* Processing state (queue depth, completion) is observable


## Transcription Models & Findings

### Model Evaluation Summary

| Model                        | Strengths                                                        | Limitations                                                                                      | Outcome             |
| ---------------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------- |
| **Google MedASR** https://huggingface.co/google/medasr            | Clinically trained, fast on CPU, promising for medical dictation | Inconsistent transcript quality in noisy environments; struggled with non-native English accents | Not adopted         |
| **Whisper `large-v3-turbo`** https://huggingface.co/openai/whisper-large-v3-turbo | Highest accuracy among tested models                             | 2–3× latency on CPU; disruptive for interactive dictation                                        | Tested, not default |
| **Whisper `small.en`** https://huggingface.co/openai/whisper-small.en       | Stable, predictable latency, near-real-time on CPU               | Slightly reduced accuracy vs larger models                                                       | **Current default** |

> Evaluation focused on **CPU-first**, **privacy-aware**, near-real-time dictation workflows (especially radiology reporting).



### Current Model Selection

**Whisper `small.en`** is used by default as a practical balance between:

* transcription quality
* operational stability
* near-real-time responsiveness on CPU

Larger Whisper variants improved accuracy but introduced unacceptable latency under CPU-only conditions for interactive use.



### CPU Configuration

When running Whisper on CPU, inference stability was best when:

* `cpu_threads` ≈ **physical CPU cores**
  (rather than logical cores / hyperthreads)

This reduced contention and latency spikes during sustained dictation.


### Configuration Note

* Model and engine configuration is currently **hardcoded**
* Located in `engine.py`
* Refactoring for improved configurability is planned


---


## Running Locally
Setup python virtual environment and packages
```bash
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

```bash
python server.py
```

Then open `client.html` in a browser and connect to:

```
http://localhost:8000
```

### Microphone Access Permission
Modern web browsers enforce strict security rules: microphone access is only allowed in secure contexts (typically HTTPS).

- **Local testing**: Works on http://localhost without issues.
- **Network testing**: Requires HTTPS. Use a self-signed SSL certificate with tools like Apache or Nginx to proxy to localhost.

**A quick Nginx HTTPS setup:**
```nginx
server {
    listen 443 ssl;
    
    ssl_certificate /path/to/self-signed.crt;
    ssl_certificate_key /path/to/self-signed.key;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```


## Known Limitations

* In-memory state only (no persistence)
* No authentication or access control
* No fault tolerance or recovery
* Polling-based transcript delivery
* Configuration is not yet externalized



## Future Directions

Possible extensions include:

* Externalized configuration (models, threading, chunking, api)
* Persistent transcript storage
* Alternative transport (e.g. WebSockets)
* Tightening security
