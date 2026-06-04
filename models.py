import torch
import nemo.collections.asr as nemo_asr
from nemo.collections.asr.models import ASRModel
from transformers.audio_utils import load_audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, CohereAsrForConditionalGeneration
from data_initiation import DS
# from qwen_asr import Qwen3ASRModel
import time


class BaseModel:
    def __init__(self, name, model, processor=None):

        # variables
        self.name = name
        self.model = model
        self.processor = processor

    def transcribe(self, audio) -> str:
        pass


# CONSTANTS
sample_rate = 16000
batch_size = 1
max_new_tokens = 256
language = 'du'
torch_type = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------- Qwen3 ASR 1.7B ----------------
# class QwenAsr(BaseModel):
#     def __init__(self):
#         super().__init__(
#                 name="Qwen3 ASR 1.7B",
#                 model=Qwen3ASRModel.from_pretrained(
#                     "Qwen/Qwen3-ASR-1.7B",
#                     dtype=torch.bfloat16,
#                     device_map=device,
#                     # attn_implementation="flash_attention_2",
#                     max_inference_batch_size=batch_size, # Batch size limit for inference. -1 means unlimited. Smaller values can help avoid OOM.
#                     max_new_tokens=max_new_tokens # Maximum number of tokens to generate. Set a larger value for long audio input.
#                 )
#         )
#
#     def transcribe(self, audio):
#         r esult = self.model.transcribe(
#             audio=audio,
#             language='Dutch'
#         )
#         return result


# --------------- Cohere Labs Transcribe ----------------
class CohereAsr(BaseModel):
    def __init__(self):
        super().__init__(
            name="Cohere Labs Transcribe",
            model=CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map=device),
            processor=AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
        )

    def transcribe(self, dataset: 'DS'):

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

                outputs = self.model.generate(**safe_inputs, max_new_tokens=256)
                text = self.processor.decode(outputs, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="nl")[0]

                duration = round((time.perf_counter() - timer) / 60, 5)
                print(f"< Transcription completed in {round(duration, 2)} minutes")

                transcripts[row['name']] = text
                times[row['name']] = duration

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

    def transcribe(self, dataset):

        transcripts = {}
        times = {}

        for row in dataset.data:
            timer = time.perf_counter()
            print(f"> Attempting to transcribe {row['name']} from dataset {dataset.name}")
            text = self.pipe(row['audio']['array'],
                             return_timestamps=True,
                             generate_kwargs={"language": "nl",
                                              "task": "transcribe"})['text']
            duration = round((time.perf_counter() - timer) / 60, 4)
            print(f"< Transcription completed in {round(duration, 2)} minutes")

            transcripts[row['name']] = text
            times[row['name']] = duration

        return transcripts, times


# --------------- NVIDIA Parakeet TDT 0.6B v3 ----------------
class ParakeetAsr(BaseModel):
    def __init__(self):
        super().__init__(
            name="NVIDIA Parakeet TDT 0.6B v3",
            model=ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        )
        self.model.change_attention_model(self_attention_model="rel_pos_local_attn", att_context_size=[256, 256])

    def transcribe(self, dataset: 'DS'):

        transcripts = {}
        times = {}

        for row in dataset.data:
            timer = time.perf_counter()
            print(f"> Attempting to transcribe {row['name']} from dataset {dataset.name}")
            audio = row['audio']['array']
            text = self.model.transcribe(audio)
            duration = round((time.perf_counter() - timer) / 60, 4)
            print(f"< Transcription completed in {round(duration, 2)} minutes")

            transcripts[row['name']] = text[0].text
            times[row['name']] = duration

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
            duration = round((time.perf_counter() - timer) / 60, 4)
            print(f"< Transcription completed in {round(duration, 2)} minutes")

            transcripts[row['name']] = text[0].text
            times[row['name']] = duration

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
