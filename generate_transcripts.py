# from models.asr_models_nemo import ParakeetAsr, CanaryAsr
import os
from pathlib import Path
from datetime import datetime
import csv
import librosa
import argparse
import gc
from models.base_model import VadModel, CONFIG


#
#
#       This file runs a list of models on a list of directories containing structured audio files
#       It outputs a single csv file as follows
#
#       - output
#       | - asr output
#         | - year-month-day_hour-min-second.csv  ->  name, category, model, time, transcript
#
#


class PipeLine1:
    def __init__(self, models, categories: list[str]):
        self.models = models
        self.dataset_paths = categories
        self.output_file_path = Path('output/asr output')

    def run(self):

        headers = ['name', 'category', 'model', 'cpu_max', 'cpu_avg', 'gpu_max', 'gpu_avg', 'mem_max', 'mem_avg', 'rtfx', 'transcript']
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        vad_model = VadModel()

        with open(self.output_file_path / Path(f"{current_time}.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for model in self.models:
                model = model()
                if model.name == "Qwen3 ASR 1.7B":
                    model.vad = vad_model
                for dataset_path in self.dataset_paths:
                    print(f"[SYS]    Processing dataset {dataset_path}")
                    audio_directory = Path("dataset") / dataset_path / Path('audio')
                    for file in os.listdir(audio_directory):
                        print(f"[SYS]    Processing audio file {file}")
                        audio, process_time = librosa.load(audio_directory / file, sr=CONFIG['sample_rate'], mono=True)

                        data_file = {
                            'name': file,
                            'category': dataset_path,
                            'audio': vad_model.get_speech_chunks(audio, CONFIG['sample_rate']) if CONFIG['use_vad'] else audio
                        }
                        if model.name == "Whisper-large-v3-fast":
                            data_file['audio'] = audio

                        transcript, process_time = model.run(data_file)
                        audio_duration = audio.shape[-1] / 16000

                        writer.writerow({
                            'name': file,
                            'category': dataset_path,
                            'transcript': transcript,
                            'model': model.name,
                            'cpu_max': summary['cpu_pct_max'],
                            'cpu_avg': summary['cpu_pct_avg'],
                            'gpu_max': summary['gpu_util_max'],
                            'gpu_avg': summary['gpu_util_avg'],
                            'mem_max': summary['gpu_mem_mb_max'],
                            'mem_avg': summary['gpu_mem_mb_avg'],
                            'rtfx': audio_duration / process_time,
                        })

                del model
                gc.collect()
                torch.cuda.empty_cache


if __name__ == '__main__':
    categories = ['Dokter Patient', 'Psychologische gespreksvoering', 'interviews']
    parser = argparse.ArgumentParser(description="Run the transcription process for all the models of a specific env")
    parser.add_argument(
        'target',
        nargs='?',
        choices=['nemo', 'latest'],
        help="Target version (Options: nemo, latest)"
    )

    env = parser.parse_args().target

    if env == 'nemo':
        try:
            from models.asr_models_nemo import ParakeetAsr, CanaryAsr
        except ImportError as e:
            print(f"[SYS] X  Environment doesn't match the function call causing an import error: {e}")

        a = PipeLine1(models=[CanaryAsr], categories=categories)

    elif env == 'latest':
        try:
            from models.asr_models_cohere import CohereAsr
        except ImportError as e:
            print(f"[SYS] X  Environment doesn't match the function call causing an import error: {e}")
        a = PipeLine1(models=[CohereAsr], categories=categories)

    else:
        try:
            from models.asr_models import WhisperAsrFast, QwenAsr
        except ImportError as e:
            print(f"[SYS] X  Environment doesn't match the function call causing an import error: {e}")
        a = PipeLine1(models=[WhisperAsrFast, QwenAsr], categories=categories)

    a.run()
