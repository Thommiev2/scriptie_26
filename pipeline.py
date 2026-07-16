import os
import csv
import argparse
import librosa
from pathlib import Path
from datetime import datetime

# Assuming you have added models/__init__.py and models/base_model.py
from models.base_model import VadModel, CONFIG

class PipeLine1:
    def __init__(self, models: list, categories: list):
        self.models = models
        self.dataset_paths = categories

        self.output_dir = Path("output/asr_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        headers = ['name', 'category', 'model', 'rtfx', 'transcript']
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        output_file = self.output_dir / f"{current_time}.csv"

        vad_model = VadModel()

        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for model_class in self.models:
                model = model_class()

                if model.name == "Qwen3 ASR 1.7B":
                    model.vad = vad_model

                for category in self.dataset_paths:
                    audio_dir = Path("dataset") / category / "audio"
                    print(f"[SYS]   Processing dataset: {category}")

                    if not audio_dir.exists():
                        print(f"[SYS] X  Directory not found: {audio_dir}")
                        continue

                    for file in os.listdir(audio_dir):

                        file_path = audio_dir / file

                        audio, _ = librosa.load(file_path, sr=CONFIG['sample_rate'], mono=True)

                        data_file = {
                            'name': file,
                            'category': category,
                            'audio': vad_model.get_speech_chunks(audio, CONFIG['sample_rate']) if CONFIG['use_vad'] else audio
                        }


                        if model.name == "Whisper-large-v3-fast":
                            data_file['audio'] = audio

                        transcript, process_time = model.run(data_file)
                        audio_duration = audio.shape[-1] / CONFIG['sample_rate']

                        writer.writerow({
                            'name': file,
                            'category': category,
                            'transcript': transcript,
                            'model': model.name,
                            'rtfx': audio_duration / process_time
                        })

if __name__ == '__main__':
    categories = ['Dokter Patient', 'Psychologische gespreksvoering', 'interviews']

    parser = argparse.ArgumentParser(description="Run ASR Pipeline")
    parser.add_argument('target', nargs='?', choices=['nemo', 'latest'], help="Environment selection")
    args = parser.parse_args()

    import_map = {
        'nemo': ('models.asr_models_nemo', ['ParakeetAsr', 'CanaryAsr']),
        'latest': ('models.asr_models_cohere', ['CohereAsr']),
        None: ('models.asr_models', ['WhisperAsrFast', 'QwenAsr'])
    }

    module_path, class_names = import_map.get(args.target, import_map[None])

    try:
        module = __import__(module_path, fromlist=class_names)
        models_to_run = [getattr(module, name) for name in class_names]

        pipeline = PipeLine1(models=models_to_run, categories=categories)
        pipeline.run()
    except Exception as e:
        print(f"[SYS] X Error initializing environment: {e}")
