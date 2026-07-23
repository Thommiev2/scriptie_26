import os
import numpy as np
import torch
if __name__ != "__main__":
    from models.base_model import BaseModel, CONFIG
else:
    from base_model import BaseModel, CONFIG
from transformers import AutoProcessor, CohereAsrForConditionalGeneration
import time
from pathlib import Path
import librosa


# --------------- Cohere Labs Transcribe ----------------
class CohereAsr(BaseModel):
    def __init__(self):
        super().__init__(
            name="Cohere Labs Transcribe",
            model=CohereAsrForConditionalGeneration.from_pretrained(
                "CohereLabs/cohere-transcribe-03-2026",
                device_map=CONFIG['device'],
                dtype=CONFIG['dtype']
            ),
            processor=AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026")
        )
        print('[CHR] v  Model initialized and loaded in succesfully')

    def transcribe(self, data_file) -> (str, float):

        if not isinstance(data_file['audio'], list):
            data_file['audio'] = [data_file['audio']]

        timer = time.perf_counter()

        print(f"[CHR] >  Attempting to transcribe {data_file['name']} from dataset {data_file['category']}")

        transcript = ''
        for chunk in data_file['audio']:
            inputs = self.processor(chunk, sampling_rate=CONFIG['sample_rate'], return_tensors='pt', language='nl')
            inputs = inputs.to(self.model.device, dtype=self.model.dtype)
            audio_chunk_index = inputs.get("audio_chunk_index")
            output = self.model.generate(**inputs, max_new_tokens=CONFIG['max_new_tokens'])
            text = self.processor.decode(output, skip_special_tokens=True, audio_chunk_index=audio_chunk_index, language="nl")[0]

            transcript += " " + text.strip()

            # Fixes problem where GPU runs out of memory to allocate the audio files to.
            del inputs, output, audio_chunk_index
            if CONFIG['device'] == 'cuda':
                torch.cuda.empty_cache()

        process_time = (time.perf_counter() - timer)
        seconds = int(process_time % 60)
        print(f"[CHR] <  Transcription completed in {int(process_time/60)}:{'0' if seconds < 10 else ''}{seconds} minutes")

        return transcript, process_time


if __name__ == "__main__":
    models = [CohereAsr]
    for model in models:
        model = model()
        model.validate_model()

# for mod in [CohereAsr, ]:
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
#         t = m.run(b)
#         print(t)

