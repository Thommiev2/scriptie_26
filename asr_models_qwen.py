import transformers
if transformers.__version__ != "4.57.6":
    raise ImportError("The models in 'asr_models_qwen' require version 4.57.6 of the transformer library.\n"
                      "Run 'python transformer_version' to install the correct version.")

import os
from qwen_asr import Qwen3ASRModel
import torch

import time
from pathlib import Path

#
#
#       Qwen3Asr uses a different version of the transformers architecture.
#
#


class BaseModel:
    def __init__(self, name, model, processor=None):

        # variables
        self.name = name
        self.model = model
        self.processor = processor

    def transcribe(self, dataset: 'DS') -> (dict[str: str], dict[str: float]):
        pass


# CONSTANTS
sample_rate = 16000
batch_size = 1
max_new_tokens = 256
language = 'du'
torch_type = torch.bfloat16 if torch.cuda.is_available() else torch.bfloat16
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------- Qwen3 ASR 1.7B ----------------
class QwenAsr(BaseModel):
    def __init__(self):

        # if transformers.__version__ == ""

        super().__init__(
                name="Qwen3 ASR 1.7B",
                model=Qwen3ASRModel.from_pretrained(
                    "Qwen/Qwen3-ASR-1.7B",
                    dtype=torch.bfloat16,
                    device_map=device,
                    max_inference_batch_size=batch_size,
                    max_new_tokens=max_new_tokens
                )
        )

    def transcribe(self, dataset: 'DS') -> (dict[str: str], dict[str: float]):

        transcripts = {}
        times = {}

        for row in dataset.data:

            for file in os.listdir(dataset.audio):
                name, f_type = file.split('.')
                if name == row['name']:
                    audio_path = str(dataset.audio / Path(f"{name}.{f_type}"))
                    break
            else:
                print(f"X ERROR CANT FIND MATCHING AUDIO FILE FOR {row['name']}")
                return None, None

            timer = time.perf_counter()
            print(f"> Attempting to transcribe {row['name']} from dataset {dataset.name}")

            with torch.no_grad():
                text = self.model.transcribe(
                    audio=audio_path,
                    language='Dutch'
                )

            process_time = (time.perf_counter() - timer)
            print(f"< Transcription completed in {int(process_time/60)}:{int(process_time % 60)} minutes")
            print(text[0])
            transcripts[row['name']] = text[0].text
            times[row['name']] = round(process_time / 60, 3)

        return transcripts, times
