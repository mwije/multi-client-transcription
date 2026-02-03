import os
import time
import csv
import platform
import psutil
import cpuinfo
import soundfile as sf
from faster_whisper import WhisperModel

# -------------------- CONFIG --------------------
MODELS = ["distil-large-v3"]
PRIMER_AUDIO = "primer.wav"
TEST_AUDIO = "test.wav"
CSV_PATH = "whisper_cpu_benchmarks2.csv"

DEVICE = "cpu"
COMPUTE_TYPE = "int8"
MAX_THREADS = os.cpu_count() or 1
# ------------------------------------------------


def audio_duration_sec(path):
    with sf.SoundFile(path) as f:
        return len(f) / f.samplerate


TEST_AUDIO_SEC = audio_duration_sec(TEST_AUDIO)


def get_hw_info():
    cpu = cpuinfo.get_cpu_info()
    mem = psutil.virtual_memory()
    return {
        "cpu_model": cpu.get("brand_raw", "unknown"),
        "cpu_arch": platform.machine(),
        "cpu_cores_logical": psutil.cpu_count(logical=True),
        "cpu_cores_physical": psutil.cpu_count(logical=False),
        "cpu_max_mhz": cpu.get("hz_advertised_friendly", "unknown"),
        "ram_gb": round(mem.total / (1024 ** 3), 2),
        "os": f"{platform.system()} {platform.release()}",
    }


def load_completed_rows():
    if not os.path.exists(CSV_PATH):
        return set()
    done = set()
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            done.add((r["model"], int(r["threads"])))
    return done


def write_row(row, fieldnames):
    exists = os.path.exists(CSV_PATH)
    with open(CSV_PATH, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def transcribe(model, audio):
    segments, _ = model.transcribe(audio, beam_size=5)
    for _ in segments:
        pass


def main():
    hw = get_hw_info()
    completed = load_completed_rows()

    fieldnames = [
        "model", "threads",
        "model_load_sec",
        "inference_sec",
        "rtf",
        "audio_sec",
        "cpu_model", "cpu_arch",
        "cpu_cores_logical", "cpu_cores_physical",
        "cpu_max_mhz", "ram_gb", "os",
    ]

    for model_name in MODELS:
        for threads in range(1, MAX_THREADS + 1):
            if (model_name, threads) in completed:
                continue

            print(f"▶ Model={model_name} | Threads={threads}")

            # ---- MODEL LOAD ----
            t0 = time.perf_counter()
            model = WhisperModel(
                model_name,
                device=DEVICE,
                compute_type=COMPUTE_TYPE,
                cpu_threads=threads,
            )
            load_time = time.perf_counter() - t0

            # ---- PRIME (not timed) ----
            transcribe(model, PRIMER_AUDIO)

            # ---- INFERENCE BENCH ----
            t1 = time.perf_counter()
            transcribe(model, TEST_AUDIO)
            infer_time = time.perf_counter() - t1

            rtf = infer_time / TEST_AUDIO_SEC

            row = {
                "model": model_name,
                "threads": threads,
                "model_load_sec": round(load_time, 4),
                "inference_sec": round(infer_time, 4),
                "rtf": round(rtf, 4),
                "audio_sec": round(TEST_AUDIO_SEC, 2),
                **hw,
            }

            write_row(row, fieldnames)
            del model


if __name__ == "__main__":
    main()
