# import transformers
# if transformers.__version__ != "4.57.6":
#     raise ImportError("The models in 'asr_models_qwen' require version 4.57.6 of the transformer library.\n"
#                       "Run 'python transformer_version' to install the correct version.")

import os
from qwen_asr import Qwen3ASRModel
from faster_whisper import WhisperModel
import torch
from base_model import BaseModel, VadModel, CONFIG
import time
from pathlib import Path

#
#
#       Qwen3Asr uses a different version of the transformers architecture.
#
#


# --------------- Qwen3 ASR 1.7B ----------------
class QwenAsr(BaseModel):
    def __init__(self):

        # if transformers.__version__ == ""

        super().__init__(
                name="Qwen3 ASR 1.7B",
                model=Qwen3ASRModel.from_pretrained(
                    "Qwen/Qwen3-ASR-1.7B",
                    dtype=CONFIG['dtype'],
                    device_map=CONFIG['device'],
                    max_inference_batch_size=4,
                    max_new_tokens=CONFIG['max_new_tokens']
                )
        )
        self.vad: VadModel | None = None
        print('[QWN] v  Model initialized and loaded in succesfully')

    def transcribe(self, data_file: dict) -> (str, float):

        timer = time.perf_counter()
        print(f"[QWN] >  Attempting to transcribe {data_file['name']} from dataset {data_file['category']}")

        chunks = self.vad.get_speech_chunks(data_file['audio'], CONFIG['sample_rate'])
        inputs = list(zip(chunks, [CONFIG['sample_rate']] * len(chunks)))

        text = self.model.transcribe(
            audio=inputs,
            language=[CONFIG['language']] * len(chunks)
        )
        process_time = (time.perf_counter() - timer)
        seconds = int(process_time % 60)
        print(f"[QWN] <  Transcription completed in {int(process_time/60)}:{'0' if seconds < 10 else ''}{seconds} minutes")

        transcript = ' '.join([t.text.strip() for t in text])

        return transcript, process_time


class WhisperAsrFast(BaseModel):
    def __init__(self):
        super().__init__(
            name="Whisper-large-v3-fast",
            model=WhisperModel(
                'large-v3',
                device=CONFIG['device'],
                # compute_type=CONFIG['dtype']
                compute_type='int8'
            )
        )
        print('[WPF] v  Model initialized and loaded in succesfully')

    def transcribe(self, data_file: dict) -> (str, float):

        timer = time.perf_counter()
        print(f"[WPF] >  Attempting to transcribe {data_file['name']} from dataset {data_file['category']}")
        text, info = self.model.transcribe(
            data_file['audio'],
            beam_size=CONFIG['beam_size']
        )
        process_time = (time.perf_counter() - timer)
        seconds = int(process_time % 60)
        print(f"[WPF] <  Transcription completed in {int(process_time/60)}:{'0' if seconds < 10 else ''}{seconds} minutes")

        transcript = " ".join([segment.text.strip() for segment in text])

        return transcript, process_time
