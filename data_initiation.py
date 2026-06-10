import os
import csv
from pathlib import Path, PurePosixPath
import librosa
import soundfile as sf

# LOAD IN AUDIO/GROUND TRUTH DATASETS THAT ARE USED FOR ASR TRANSCRIPTION


class DS:
    def __init__(self, category):
        self.name = category
        self.audio = Path("dataset") / category / "audio"
        self.ground_truth = Path("dataset") / category / "ground truth"

        self.duration = 0
        self.data = self.load_data()

    def load_data(self):
        files = list(self.audio.glob("*"))

        dataset = []
        total_duration = 0.0

        for file_path in files:
            audio, sr = sf.read(file_path)
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)

            duration = len(audio) / 16000
            total_duration += duration

            dataset.append({
                "path": str(file_path),
                "audio": audio,
                "sampling_rate": sr,
                "duration": duration
            })

        print(f"Total duration of '{self.name}' audio files: {int(total_duration / 60)}:{int(total_duration % 60)} minutes")

        self.duration = total_duration
        return dataset



# for dataset in datasets:
#     print(dataset.audio, dataset.category, dataset.name)


# fleurs = load_dataset("google/fleurs", "nl_nl", split="train")
# print(fleurs)
# audio = fleurs[0]['audio']
# transcript = fleurs[0]['transcription']
# r_transcription = fleurs[0]['raw_transcription']
# print(transcript)
# print(r_transcription)

# a = DS("Test")
# Dataset("Pedagogische gesprekken")


# path = Path(r'C:\Users\thoma\PycharmProjects\Scriptie_26\dataset\Dokter Patient')
# print(path) ssh -i ~/Downloads/vm-poc-tenant-expermimentation_key.pem azure@4.180.21.26
# b = str(PurePosixPath(path))
# print(b[:2] + b[3:])

# path = Path('dataset')
# for category in os.listdir(path):
#     for type in ['audio', 'ground truth']:
#         n_path = path / category / type
#         for file in os.listdir(n_path):
#             os.rename(n_path / file, n_path / Path(f"{file.replace('_', '-').split(' ')[0]}.{file.split('.')[-1]}"))
