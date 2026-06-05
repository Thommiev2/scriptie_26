import os
import csv
from pathlib import Path, PurePosixPath
from datasets import load_dataset, Audio, get_dataset_config_names, Value, Features, Dataset

# LOAD IN AUDIO/GROUND TRUTH DATASETS THAT ARE USED FOR ASR TRANSCRIPTION
#
# - dataset
# | - [category]
#   | - audio
#     | - [name] audio.[mp3/wav]
#   | - ground truth
#     | - [name] transcript.txt


class DS:
    def __init__(self, category):
        self.name = category
        self.audio = Path("dataset") / category / Path("audio")
        self.ground_truth = Path("dataset") / category / Path("ground truth")
        # self.clean = clean_func[category]
        self.duration = 0
        self.data = self.load_data()

    def load_data(self):
        dataset = load_dataset("audiofolder", data_dir=str(self.audio), split="train", streaming=True)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

        print(f'First row of dataset {self.name}: {next(iter(dataset))}')

        total_duration = 0
        for file in dataset["audio"]:
            print(file)
            total_duration += len(file["array"]) / file["sampling_rate"]

        print(f"Total duration of '{self.name}' audio files: {round(total_duration / 60, 1)} minutes")
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

# a = DS("Dokter Patient")
# Dataset("Pedagogische gesprekken")


# path = Path(r'C:\Users\thoma\PycharmProjects\Scriptie_26\dataset\Dokter Patient')
# print(path)
# b = str(PurePosixPath(path))
# print(b[:2] + b[3:])

