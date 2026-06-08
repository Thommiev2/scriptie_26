import transformers
if transformers.__version__ != "5.9.0":
    raise ImportError("The models in asr_models require version 5.9.0 of the transformer library.\n"
                      "Run 'python transformer_version --upgrade' to install the correct version.")

import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel
from transformers.audio_utils import load_audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, CohereAsrForConditionalGeneration
from data_initiation import DS
import time
from qwen_asr import Qwen3ASRModel


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
torch_type = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------- Cohere Labs Transcribe ----------------
class CohereAsr(BaseModel):
    def __init__(self):
        super().__init__(
            name="Cohere Labs Transcribe",
            model=CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map=device),
            processor=AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
        )

    def transcribe(self, dataset: 'DS') -> (dict[str: str], dict[str: float]):

        transcripts = {}
        times = {}

        for row in dataset.data:
            timer = time.perf_counter()
            print(f"> Attempting to transcribe {row['name']} from dataset {dataset.name}")
            audio = row['audio']['array']
            inputs = self.processor(audio, sampling_rate=row['audio']['sampling_rate'], return_tensors='pt', language='nl')
            audio_chunk_index = inputs.get("audio_chunk_index")
            with torch.no_grad():
                safe_inputs = {}
                for k, v in inputs.items():
                    if torch.is_tensor(v):
                        if torch.is_floating_point(v):
                            safe_inputs[k] = v.to(self.model.device, dtype=self.model.dtype)
                        else:
                            safe_inputs[k] = v.to(self.model.device)
                    else:
                        safe_inputs[k] = v

                output = self.model.generate(**safe_inputs, max_new_tokens=256)
                text = self.processor.decode(output, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="nl")[0]

                process_time = (time.perf_counter() - timer)
                print(f"< Transcription completed in {int(process_time/60)}:{int(process_time % 60)} minutes")

                transcripts[row['name']] = text
                times[row['name']] = round(process_time/60, 3)

        return transcripts, times


# --------------- Whisper-large-v3 ----------------
class WhisperAsr(BaseModel):
    def __init__(self):
        super().__init__(
            name="Whisper-large-v3",
            model=AutoModelForSpeechSeq2Seq.from_pretrained(
                "openai/whisper-large-v3",
                dtype=torch_type,
                low_cpu_mem_usage=True
            ),
            processor=AutoProcessor.from_pretrained("openai/whisper-large-v3")
        )

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            chunk_length_s=30,
            batch_size=batch_size,
            torch_dtype=torch_type,
            device=device,
        )

        # print(self.pipe.model.generation_config)

    def transcribe(self, dataset) -> (dict[str: str], dict[str: float]):

        transcripts = {}
        times = {}

        for row in dataset.data:
            timer = time.perf_counter()
            print(f"> Attempting to transcribe {row['name']} from dataset {dataset.name}")
            text = self.pipe(row['audio']['array'],
                             return_timestamps=True,
                             generate_kwargs={"language": "nl",
                                              "task": "transcribe"})['text']
            process_time = (time.perf_counter() - timer)
            print(f"< Transcription completed in {int(process_time/60)}:{int(process_time % 60)} minutes")

            transcripts[row['name']] = text
            times[row['name']] = round(process_time/60, 3)

        return transcripts, times


# --------------- NVIDIA Parakeet TDT 0.6B v3 ----------------
class ParakeetAsr(BaseModel):
    def __init__(self):
        super().__init__(
            name="NVIDIA Parakeet TDT 0.6B v3",
            model=ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        )
        self.model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])

    def transcribe(self, dataset: 'DS') -> (dict[str: str], dict[str: float]):

        transcripts = {}
        times = {}

        for row in dataset.data:
            timer = time.perf_counter()
            audio = row['audio']['array']

            print(f"> Attempting to transcribe {row['name']} from dataset {dataset.name}")
            text = self.model.transcribe(audio)

            process_time = (time.perf_counter() - timer)
            print(f"< Transcription completed in {int(process_time/60)}:{int(process_time % 60)} minutes")

            transcripts[row['name']] = text[0].text
            times[row['name']] = round(process_time/60, 3)

        return transcripts, times


# --------------- NVIDIA Canary 1B v2 ----------------
class CanaryAsr(BaseModel):
    def __init__(self):
        super().__init__(
            name="NVIDIA Canary 1B v2",
            model=ASRModel.from_pretrained(model_name="nvidia/canary-1b-v2")
        )

    def transcribe(self, dataset: 'DS'):

        transcripts = {}
        times = {}

        for row in dataset.data:
            timer = time.perf_counter()
            print(f"> Attempting to transcribe {row['name']} from dataset {dataset.name}")
            audio = row['audio']['array']
            text = self.model.transcribe(audio, source_lang='nl', target_lang='nl')

            process_time = (time.perf_counter() - timer)
            print(f"< Transcription completed in {int(process_time/60)}:{int(process_time % 60)} minutes")

            transcripts[row['name']] = text[0].text
            times[row['name']] = round(process_time/60, 3)

        return transcripts, times

# ----------------------- mms-1b-all -----------------------
# class MmsAllAsr(AsrModel):
#     def __init__(self):
#         super().__init__(
#             name="mms-1b-all",
#             model=
#         )


# for mod in [CanaryAsr]:
#     m = mod()
#     print(m.name)
#     output = m.transcribe(DS("Test"))
#     print(output)
