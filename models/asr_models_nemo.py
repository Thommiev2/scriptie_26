# import transformers
# if transformers.__version__ != "5.9.0":
#     raise ImportError("The models in asr_models require version 5.9.0 of the transformer library.\n"
#                       "Run 'python transformer_version --upgrade' to install the correct version.")

import os

import numpy as np
import torch
from nemo.collections.asr.models import ASRModel
from base_model import BaseModel, CONFIG
import librosa
import time
from pathlib import Path


# # --------------- Whisper-large-v3 ----------------
# class WhisperAsr(BaseModel):
#     def __init__(self):
#
#         super().__init__(
#             name="Whisper-large-v3",
#             model=AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3",
#                                                             low_cpu_mem_usage=True),
#             processor=AutoProcessor.from_pretrained("openai/whisper-large-v3")
#         )
#
#         self.pipe = pipeline(
#             "automatic-speech-recognition",
#             model=self.model,
#             tokenizer=self.processor.tokenizer,
#             feature_extractor=self.processor.feature_extractor,
#             # chunk_length_s=30,
#             batch_size=CONFIG['number_of_batches'],
#             dtype=CONFIG['dtype'],
#             device=CONFIG['device'],
#         )
#
#         print(self.pipe.model.generation_config)
#
#     def transcribe(self, data_file) -> (dict[str: str], dict[str: float]):
#
#         timer = time.perf_counter()
#
#         print(f"> Attempting to transcribe {data_file['name']} from dataset {data_file['category']}")
#         transcript = self.pipe(data_file['audio'],
#                                generate_kwargs={"language": "nl",
#                                                 "task": "transcribe"})['text']
#         process_time = (time.perf_counter() - timer)
#         print(f"< Transcription completed in {int(process_time/60)}:{int(process_time % 60)} minutes")
#
#         return transcript, process_time


# --------------- NVIDIA Parakeet TDT 0.6B v3 ----------------
class ParakeetAsr(BaseModel):
    def __init__(self):
        super().__init__(
            name="NVIDIA Parakeet TDT 0.6B v3",
            model=ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        )
        self.model.change_attention_model("rel_pos_local_attn", [128, 128])
        self.model.change_subsampling_conv_chunking_factor(1)

        print('[PRK] v  Model initialized and loaded in succesfully')

    def transcribe(self, data_file: dict) -> (str, float):

        timer = time.perf_counter()

        print(f"[PRK] >  Attempting to transcribe {data_file['name']} from dataset {data_file['category']}")
        text = self.model.transcribe(data_file['audio'])
        process_time = (time.perf_counter() - timer)
        seconds = int(process_time % 60)
        print(f"[PRK] <  Transcription completed in {int(process_time/60)}:{'0' if seconds < 10 else ''}{seconds} minutes")

        transcript = text[0].text

        return transcript, process_time


# --------------- NVIDIA Canary 1B v2 ----------------
class CanaryAsr(BaseModel):
    def __init__(self):
        super().__init__(
            name="NVIDIA Canary 1B v2",
            model=ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
        )
        print('[CNR] v  Model initialized and loaded in succesfully')

    def transcribe(self, data_file: dict) -> (str, float):

        if isinstance(data_file['audio'], list):
            data_file['audio'] = np.concatenate(data_file['audio'])

        timer = time.perf_counter()

        print(f"[CNR] >  Attempting to transcribe {data_file['name']} from dataset {data_file['category']}")
        text = self.model.transcribe(data_file['audio'], source_lang='nl', target_lang='nl', batch_size=1)
        process_time = (time.perf_counter() - timer)
        seconds = int(process_time % 60)
        print(f"[CNR] <  Transcription completed in {int(process_time/60)}:{'0' if seconds < 10 else ''}{seconds} minutes")

        transcript = text[0].text

        return transcript, process_time


# ----------------------- mms-1b-all -----------------------
# class MmsAllAsr(AsrModel):
#     def __init__(self):
#         super().__init__(
#             name="mms-1b-all",
#             model=
#         )

if __name__ == "__main__":
    models = [ParakeetAsr, CanaryAsr]
    for model in models:
        model = model()
        model.validate_model()
    # for mod in [ParakeetAsr, CanaryAsr]:
    #     m = mod()
    #     print(m.name)
    #     p = Path("dataset/Test/audio")
    #
    #     for file in os.listdir(p):
    #         audio, _ = librosa.load(p / file, sr=16000, mono=True)
    #         b = {
    #             'name': file,
    #             'category': 'Test',
    #             'audio': audio
    #         }
    #
    #         text = m.run(b)
    #         print(text)

    # c, _ = librosa.load('test.wav', sr=16000, mono=True)
    # b = {
    #     'name': 'test.wav',
    #     'category': 'testing',
    #     'audio': c
    # }
    #
    # a = ParakeetAsr()
    # d = a.transcribe(b)
    # print(d)
