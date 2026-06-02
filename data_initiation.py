import os
from pathlib import Path
from datasets import load_dataset


datasets = []


class Dataset:
    def __init__(self, audio, ground_truth, category, name):
        self.audio = audio
        self.ground_truth = ground_truth
        self.category = category
        self.name = name


def dok_pat(t):
    t = t.split('\n')
    t = " ".join([t[i*2+1] for i in range(int(len(t)/2))])
    return t


def ped_ges(t):
    t = t.split('\n')
    st = ''
    for line in t:
        st += line[line.find(':')+1:]
    return st


clean = {'Dokter Patient': dok_pat, 'Pedagogische gesprekken': ped_ges}
type = {'Dokter Patient': 'mp3', 'Pedagogische gesprekken': 'mp3'}

for category in os.listdir('dataset'):
    p = Path('dataset') / category / Path('ground truth')
    for text in os.listdir(p):
        with open(p / text, encoding='UTF-8') as f:
            ground_truth = f.read()
            ground_truth = clean[category](ground_truth)
            datasets.append(Dataset(audio=Path('dataset') / category / Path(f"audio/{text[:text.rfind(' ')]} audio.{type[category]}"),
                                    ground_truth=ground_truth,
                                    category=category,
                                    name=text[:text.rfind(' ')]))

# for dataset in datasets:
#     print(dataset.audio, dataset.category, dataset.name)


fleurs = load_dataset("google/fleurs", "nl_nl", split="train")
print(fleurs)
audio = fleurs[0]['audio']
transcript = fleurs[0]['transcription']
r_transcription = fleurs[0]['raw_transcription']
print(transcript)
print(r_transcription)
