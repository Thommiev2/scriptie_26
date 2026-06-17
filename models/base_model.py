import torch
import numpy as np
import librosa


class BaseModel:
    def __init__(self, name, model, processor=None):

        # variables
        self.name = name
        self.model = model
        self.processor = processor

        if isinstance(self.model, torch.nn.Module):
            self.model.eval()

    def run(self, data_file):

        if isinstance(self.model, torch.nn.Module):
            with torch.inference_mode():
                return self.transcribe(data_file)
        data = self.transcribe(data_file)
        return data[0].strip(), data[1]

    def transcribe(self, data_file: dict) -> (str, float):
        return '', 0

    def validate_model(self) -> bool:

        print(f"[SYS]    Validating {self.name}")

        audio, process_time = librosa.load("../test.wav", sr=16000, mono=True)

        vad_model = VadModel() if CONFIG['use_vad'] else None
        vad_audio = vad_model.get_speech_chunks(audio, CONFIG['sample_rate']) if CONFIG['use_vad'] else audio

        data_file = {
             'name': 'test',
             'category': 'test',
             'audio': audio if self.model == "Whisper-large-v3-fast" else vad_audio
        }

        try:
            text, time = self.run(data_file)
            if not isinstance(text, str):
                print(f"[SYS] X  Transcript ouput of model {self.name} is not of type str")
            if not isinstance(time, float):
                print(f"[SYS] X  Process time ouput of model {self.name} is not of type float")
        except Exception as e:
            print(f"[SYS] X  The following error occured while running model {self.name}\n{e}")
            return False

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

