import torch
import numpy as np
import librosa
import threading
import time
import psutil
import pynvml


class BaseModel:
    def __init__(self, name, model, processor=None):

        # variables
        self.name = name
        self.model = model
        self.processor = processor

        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

    def run(self, data_file):
        with ResourceMonitor(interval=0.1) as mon:
            if isinstance(self.model, torch.nn.Module):
                with torch.inference_mode():
                    data = self.transcribe(data_file)
            else:
                data = self.transcribe(data_file)
        summary = mon.summary()
        return data[0].strip(), data[1], summary

    def transcribe(self, data_file: dict) -> (str, float):
        return '', 0

    def validate_model(self) -> bool:

        print(f"[SYS]    Validating {self.name}")

        audio, process_time = librosa.load("test.wav", sr=16000, mono=True)

        vad_model = VadModel() if CONFIG['use_vad'] else None
        vad_audio = vad_model.get_speech_chunks(audio, CONFIG['sample_rate']) if CONFIG['use_vad'] else audio

        data_file = {
             'name': 'test',
             'category': 'test',
             'audio': audio if self.model == "Whisper-large-v3-fast" else vad_audio
        }

        try:
            text, process_time, summary = self.run(data_file)
            if not isinstance(summary, dict):
                print(f"[SYS] X  Summary of gpu, cpu and memory usage of model {self.name} is not of type dict")
            if not isinstance(text, str):
                print(f"[SYS] X  Transcript ouput of model {self.name} is not of type str")
            if not isinstance(time, float):
                print(f"[SYS] X  Process time ouput of model {self.name} is not of type float")
        except Exception as e:
            print(f"[SYS] X  The following error occured while running model {self.name}\n{e}")
            return False

        print(f"[SYS] <  Summary output {summary}")
        print(f"[SYS] v  Validating model {self.name} was succesfull")

        return True


class VadModel:
    def __init__(self):
        self.model = None
        self.timestamps = None  # get_speech_timestamps
        self.load_vad_model()

    def load_vad_model(self):
        vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
        )
        self.model = vad_model
        self.get_timestamps = utils[0]  # get_speech_timestamps
        print('[VAD] v  Model initialized and loaded in succesfully')

    def filter_silence(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:

        print(f"[VAD] >  Attempting to filter empty segments")

        audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))

        speech_timestamps = self.get_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=sample_rate,
            threshold=CONFIG["audio_threshold"],
            min_speech_duration_ms=CONFIG["min_speech_duration_ms"],
            min_silence_duration_ms=CONFIG["min_silence_duration_ms"],
            speech_pad_ms=CONFIG["speech_pad_ms"],
            return_seconds=False,
        )

        if not speech_timestamps:
            print("[VAD] X  No speech detected, returning original audio")
            return audio

        segments = [
            audio_tensor[ts["start"]:ts["end"]] for ts in speech_timestamps
        ]

        filtered = torch.cat(segments).numpy()

        removed = len(audio) - len(filtered)
        print(
            f"[VAD] <  Trimmed {round(removed / sample_rate, 2)}s of non-speech and kept {len(segments)} segments"
        )

        return filtered

    def get_speech_chunks(self, audio, sampling_rate, max_chunk_duration_s=45):

        audio_tensor = torch.from_numpy(np.asarray(audio, dtype=np.float32))

        speech_timestamps = self.get_timestamps(
            audio_tensor,
            self.model,
            sampling_rate=sampling_rate,
            threshold=CONFIG["audio_threshold"],
            min_speech_duration_ms=CONFIG["min_speech_duration_ms"],
            min_silence_duration_ms=CONFIG["min_silence_duration_ms"],
            speech_pad_ms=CONFIG["speech_pad_ms"],
            return_seconds=False,
        )

        max_samples = int(max_chunk_duration_s * sampling_rate)
        chunks = []
        current_start = None
        current_end = None

        for ts in speech_timestamps:
            if current_start is None:
                current_start, current_end = ts["start"], ts["end"]
            elif ts["end"] - current_start <= max_samples:
                current_end = ts["end"]
            else:
                chunks.append(audio_tensor[current_start:current_end].numpy())
                current_start, current_end = ts["start"], ts["end"]

        if current_start is not None:
            chunks.append(audio_tensor[current_start:current_end].numpy())
        print(len(chunks))
        for chunk in chunks:
            print(len(chunk))
        return chunks if chunks else [audio]


pynvml.nvmlInit()
GPU_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(0)  # adjust index if multi-GPU


class ResourceMonitor:
    """Samples CPU/GPU usage in a background thread while a block runs."""

    def __init__(self, interval=0.1):
        self.interval = interval
        self.stop = threading.Event()
        self.samples = []

    def sample_loop(self):
        while not self.stop.is_set():
            util = pynvml.nvmlDeviceGetUtilizationRates(GPU_HANDLE)
            mem = pynvml.nvmlDeviceGetMemoryInfo(GPU_HANDLE)
            self.samples.append({
                "cpu_pct": psutil.cpu_percent(),
                "gpu_util_pct": util.gpu,
                "gpu_mem_used_mb": mem.used / 1024**2,
            })
            time.sleep(self.interval)

    def __enter__(self):
        self.thread = threading.Thread(target=self.sample_loop, daemon=True)
        self.stop.clear()
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        self.thread.join()

    def summary(self):
        if not self.samples:
            return {}
        cpu = [s["cpu_pct"] for s in self.samples]
        gpu = [s["gpu_util_pct"] for s in self.samples]
        mem = [s["gpu_mem_used_mb"] for s in self.samples]
        return {
            "cpu_pct_avg": sum(cpu) / len(cpu),
            "cpu_pct_max": max(cpu),
            "gpu_util_avg": sum(gpu) / len(gpu),
            "gpu_util_max": max(gpu),
            "gpu_mem_mb_avg": sum(mem) / len(mem),
            "gpu_mem_mb_max": max(mem),
            "n_samples": len(self.samples),
        }


CPU_CONSTANTS = {
    'device': 'cpu',
    'dtype': torch.float32
}
GPU_CONSTANTS = {
    'device': 'cuda',
    'dtype': torch.float16
}

# CONFIGURATIONS

DEVICE_CONFIG = GPU_CONSTANTS if torch.cuda.is_available() else CPU_CONSTANTS

ENCODE_CONFIG = {
    'use_vad': True,
    'silence_padding': 200,
    'sample_rate': 16000,
    'audio_threshold': 0.5,
    'speech_pad_ms': 30,
    'min_speech_duration_ms': 250,
    'min_silence_duration_ms': 100,
    'language': 'Dutch'
}

DECODE_CONFIG = {
    'beam_search': False,
    'gready_search': True,
    'beam_size': 1,
    'batch_size': 2,
    'max_new_tokens': 256,
}

CONFIG = DECODE_CONFIG | ENCODE_CONFIG | DEVICE_CONFIG

