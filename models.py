import torch
import nemo.collections.asr as nemo_asr
from transformers.audio_utils import load_audio
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline, CohereAsrForConditionalGeneration
# from qwen_asr import Qwen3ASRModel


class AsrModel:
    def __init__(self, name, model, processor=None):

        # variables
        self.name = name
        self.model = model
        self.processor = processor

    def transcribe(self, audio) -> str:
        pass


# CONSTANTS
sample_rate = 16000
batch_size = 32
max_new_tokens = 256
language = 'du'
torch_type = torch.float16 if torch.cuda.is_available() else torch.float32
device = "cuda:0" if torch.cuda.is_available() else "cpu"


# --------------- Qwen3 ASR 1.7B ----------------
# class QwenAsr(AsrModel):
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
class CohereAsr(AsrModel):
    def __init__(self):
        super().__init__(
            name="Cohere Labs Transcribe",
            model=CohereAsrForConditionalGeneration.from_pretrained("CohereLabs/cohere-transcribe-03-2026", device_map="auto"),
            processor=AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026"),
        )

    def transcribe(self, audio):
        audio = load_audio(audio, sampling_rate=sample_rate)
        inputs = self.processor(audio, sampling_rate=sample_rate, return_tensors="pt", language='nl')
        inputs.to(self.model.device, dtype=self.model.dtype)

        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        text = self.processor.decode(outputs, skip_special_tokens=False)
        return text


# --------------- Whisper-large-v3 ----------------
class WhisperAsr(AsrModel):
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
            torch_dtype=torch_type,
            device=device,
        )

    def transcribe(self, audio):
        return self.pipe(audio, generate_kwargs={"language": "dutch", "task": "transcribe"})


# --------------- NVIDIA Parakeet TDT 0.6B v3 ----------------
class ParakeetAsr(AsrModel):
    def __init__(self):
        super().__init__(
            name="NVIDIA Parakeet TDT 0.6B v3",
            model=nemo_asr.models.ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        )

    def transcribe(self, audio):
        return self.model.transcribe(audio, source_lang='nl', target_lang='nl')


# --------------- NVIDIA Canary 1B v2 ----------------
class CanaryAsr(AsrModel):
    def __init__(self):
        super().__init__(
            name="NVIDIA Canary 1B v2",
            model=nemo_asr.models.ASRModel.from_pretrained(
                model_name="nvidia/parakeet-tdt-0.6b-v3")
        )

    def transcribe(self, audio):
        return self.model.transcribe(audio, source_lang='nl', target_lang='nl')


# ----------------------- mms-1b-all -----------------------
# class MmsAllAsr(AsrModel):
#     def __init__(self):
#         super().__init__(
#             name="mms-1b-all",
#             model=
#         )


for model in [CanaryAsr, ParakeetAsr, WhisperAsr, CohereAsr]:
    m = model()
    print(m.name)
    output = m.transcribe('2086-149220-0033.wav')
    print(output)
