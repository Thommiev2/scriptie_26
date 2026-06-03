import os
from pathlib import Path, PurePosixPath
from datasets import load_dataset, Audio, get_dataset_config_names

# LOAD IN AUDIO/GROUND TRUTH DATASETS THAT ARE USED FOR ASR TRANSCRIPTION
#
# - dataset
# | - [category]
#   | - audio
#     | - [name] audio.[mp3/wav]
#   | - ground truth
#     | - [name] transcript.txt


# Clean data for category Dokter Patient directory
def dok_pat(t):
    t = t.split('\n')
    t = " ".join([t[i*2+1] for i in range(int(len(t)/2))])
    return t


# Clean data for category Pedagogische gesprekken
def ped_ges(t):
    t = t.split('\n')
    st = ''
    for line in t:
        st += line[line.find(':')+1:]
    return st


clean_func = {'Dokter Patient': dok_pat, 'Pedagogische gesprekken': ped_ges}
audio_type = {'Dokter Patient': 'mp3', 'Pedagogische gesprekken': 'mp3'}


class Dataset:
    def __init__(self, category):
        self.name = category
        self.audio = Path("dataset") / category / Path("audio")
        self.ground_truth = Path("dataset") / category / Path("ground truth")
        self.clean = clean_func[category]
        self.duration = 0
        self.data = self.load_data()

    def load_data(self):
        dataset = load_dataset("audiofolder", data_dir=str(self.audio), split="train", streaming=True)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
        sample = next(iter(dataset))
        print(sample)

        total_duration = 0
        for file in dataset["audio"]:
            total_duration += len(file["array"]) / file["sampling_rate"]

        print(f"Total duration of '{self.name}' audio files: {round(total_duration / 60, 1)} minutes")
        self.duration = total_duration

        return dataset

# for category in os.listdir('dataset'):
#     p = Path('dataset') / category / Path('ground truth')
#     for text in os.listdir(p):
#         with open(p / text, encoding='UTF-8') as f:
#             ground_truth = f.read()
#             ground_truth = clean[category](ground_truth)
#             datasets.append(Dataset(audio=Path('dataset') / category / Path(f"audio/{text[:text.rfind(' ')]} audio.{type[category]}"),
#                                     ground_truth=ground_truth,
#                                     category=category,
#                                     name=text[:text.rfind(' ')]))

# for dataset in datasets:
#     print(dataset.audio, dataset.category, dataset.name)


# fleurs = load_dataset("google/fleurs", "nl_nl", split="train")
# print(fleurs)
# audio = fleurs[0]['audio']
# transcript = fleurs[0]['transcription']
# r_transcription = fleurs[0]['raw_transcription']
# print(transcript)
# print(r_transcription)

# Dataset("Dokter Patient")
# Dataset("Pedagogische gesprekken")
path = Path(r'C:\Users\thoma\PycharmProjects\Scriptie_26\dataset\Dokter Patient')
print(path)
b = str(PurePosixPath(path))
print(b[:2] + b[3:])

