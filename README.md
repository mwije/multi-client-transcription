# 🎙️ Local Multi-Client Transcription Service

A local, session-based transcription service designed to support **multiple concurrent clients** using a shared speech-to-text engine, with particular attention to privacy, constrained compute environments, and clinical dictation workflows.

>This repository hosts an **exploratory implementation** for evaluating on-premise transcription for healthcare use cases (e.g. radiology reporting) in a resource constrained setting.

## Overview

Many healthcare institutions face dilemmas opting in for clinical dictation in healthcare workflows (e.g. radiology, clinic notes)
* Cloud-based transcription raises privacy, governance, and cost concerns
* On-premise healthcare IT infrastructure are often constrained
* Clinicians expect incremental feedback, not long batch delays

This project explores whether a **locally hosted transcription service**, shared across users, can provide acceptable latency and reliability under these constraints.



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
* **CPU-first**: designed and tested under CPU-constrained conditions. Can also be configured for GPU if available.
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
│ • Speech Recognition      │
│ • Serves across sessions  │
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

## Transcription Model Benchmarks

This project targets **CPU-only, privacy-aware, near-real-time dictation**, evaluated using a realistic clinical audio sample (≈151s, non-native speaker).

Model selection is driven by **measured latency, scaling behavior, and stability**, not peak accuracy alone.

### Models evaluated

#### [**Google MedASR**](https://huggingface.co/google/medasr)
  Clinically trained and fast on CPU, but transcript quality was low, probably due to ambient noise and non-native accents.  
  → Not adopted.

#### [**OpenAI Whisper `small.en`**](https://huggingface.co/openai/whisper-small.en)
  Consistently achieved the lowest real-time factor (RTF) across both systems.  
  Optimal performance occurred near physical core count, with clear degradation when oversubscribing threads.  
  → **Default model**

#### [**OpenAI Whisper `medium.en`**](https://huggingface.co/openai/whisper-medium.en)  
  Improved accuracy, but showed diminishing returns and unstable scaling under CPU-only execution.  
  Oversubscription significantly increased latency variance.  
  → Plausible but heavier for interactive use

#### [**Distil Whisper `distil-large-v3`**](https://huggingface.co/distil-whisper/distil-large-v3)
  Demonstrated a strong accuracy/RTF trade-off on server-class CPUs.  
  Outperformed `medium.en` in sustained throughput when threads were capped conservatively.  
  → Viable option for non-interactive or server-side workloads

#### [**OpenAI Whisper `large-v3-turbo`**](https://huggingface.co/openai/whisper-large-v3-turbo)
  Evaluated experimentally on the laptop system.
  Despite higher accuracy, CPU latency remained high for real-time dictation.  
  → Tested for insight only

#### Benchmark Insights
- Best performance occurred at or below **physical core count**
- Oversubscribing threads degrades Real-Time Factor
- Larger models were more sensitive to thread misconfiguration
![RTF vs Thread Count](benchmarks/data/rtf_vs_threads.png)
> Full benchmark data, methodology and plots are available in [`benchmarks/`](./benchmarks).



### Configuration note

Model and engine configuration is currently **hardcoded** in `engine.py`.  
This reflects the narrow operating envelope identified in benchmarking.  
Refactoring for runtime configurability is planned.

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

Then open `web client` in a browser:

```
http://localhost:8000/client.html
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
