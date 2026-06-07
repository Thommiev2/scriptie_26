import os
import csv
from pathlib import Path, PurePosixPath
from datasets import load_dataset, Audio, get_dataset_config_names, Value, Features, Dataset

# LOAD IN AUDIO/GROUND TRUTH DATASETS THAT ARE USED FOR ASR TRANSCRIPTION
# FORMATTING CHECKED PER CATEGORY BY VALIDATE_DATA_FORMATTING
#
#       - dataset
#       | - [category]
#         | - audio
#         | | - metadata.csv
#         | | - [name].[mp3/wav]
#         | - ground truth
#           | - [name].txt
#
#       Output file of DS load_data takes the same fieldnames as in the csv file (file_name, transcript, name)
#
#


class DS:
    def __init__(self, category):
        self.name = category
        self.audio = Path("dataset") / category / Path("audio")
        self.ground_truth = Path("dataset") / category / Path("ground truth")
        # self.clean = clean_func[category]
        self.duration = 0
        self.data = self.load_data()

    def load_data(self):

        validate_data_formatting(self.name)

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


def validate_data_formatting(directory):

    path = Path('dataset')
    if not path.exists():
        raise FileNotFoundError('Missing root directory for all data')
    if not (path / directory).exists():
        raise FileNotFoundError(f"Provided category '{directory}' of data is not present inside the dataset directory")
    path = path / directory
    audio = path / Path('audio')
    ground_truth = path / Path('ground truth')
    if not audio.exists():
        raise FileNotFoundError(f"Category '{directory}' does not contain a dedicated audio directory")
    if not ground_truth.exists():
        raise FileNotFoundError(f"Category '{directory}' does not contain a dedicated ground truth directory")

    gt_files = os.listdir(ground_truth)
    gt_names = [file.split('.')[0] for file in gt_files]
    gt_extensions = [file.split('.')[-1] for file in gt_files]

    meta_data_present = False

    for file in os.listdir(audio):
        if file == 'metadata.csv':
            meta_data_present = True
            continue

        name = file.split('.')[0]
        extension = file.split('.')[-1]

        if extension not in ['mp3', 'wav']:
            raise TypeError(f"file '{file}' should be of type .mp3 or .wav in category '{directory}'s audio directory")
        if '_' in file:
            raise NameError(f"file '{file}' contains the invalid character '_'")

        if name not in gt_names:
            raise FileNotFoundError(f"file '{file}' had no matching ground truth file")

    if not meta_data_present:
        raise FileNotFoundError(f"no metadata present in audio directory of category '{directory}'")

    for i in range(len(gt_extensions)):
        if gt_extensions[i] != 'txt':
            raise TypeError(f"file {gt_files[i]} should be of type .txt in category '{directory}'s audio directory")



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
