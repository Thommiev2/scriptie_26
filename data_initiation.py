import os
import csv
from pathlib import Path, PurePosixPath
import librosa
# LOAD IN AUDIO/GROUND TRUTH DATASETS THAT ARE USED FOR ASR TRANSCRIPTION


# class DataFile:
#     def __init__(self, name, category):
#         self.name = name
#         self.category = category
#
#         self.audio = audio_tensor
#         self.duration = len(audio_tensor.shape[-1] / 16000)
#
#
# def load_data(category):
#     total_duration = 0
#     data_set = {}
#     for file_name in os.listdir(Path('dataset') / category / Path('audio')):
#         audio_tensor, _ = librosa.load(str(Path("dataset") / category / Path("audio") / file_name), sr=16000, mono=True)
#         total_duration += len(audio_tensor)


# class DS:
#     def __init__(self, category):
#         self.name = category
#         self.audio = Path("dataset") / category / Path("audio")
#         self.ground_truth = Path("dataset") / category / Path("ground truth")
#         # self.clean = clean_func[category]
#         self.duration = 0
#         self.data = self.load_data()
#
#     def load_data(self):
#
#         dataset = load_dataset("audiofolder", data_dir=str(self.audio), split="train", streaming=True)
#         dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
#
#         print(f'First row of dataset {self.name}: {next(iter(dataset))}')
#
#         total_duration = 0
#         for file in dataset["audio"]:
#             print(file)
#             total_duration += len(file["array"]) / file["sampling_rate"]
#
#         print(f"Total duration of '{self.name}' audio files: {round(total_duration / 60, 1)} minutes")
#         self.duration = total_duration
#
#         return dataset





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
# print(path)
# b = str(PurePosixPath(path))
# print(b[:2] + b[3:])

# path = Path('dataset')
# for category in os.listdir(path):
#     for type in ['audio', 'ground truth']:
#         n_path = path / category / type
#         for file in os.listdir(n_path):
#             os.rename(n_path / file, n_path / Path(f"{file.replace('_', '-').split(' ')[0]}.{file.split('.')[-1]}"))
