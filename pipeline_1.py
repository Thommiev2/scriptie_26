from data_initiation import DS
from models import BaseModel, CanaryAsr, WhisperAsr, ParakeetAsr, CohereAsr
# from models import BaseModel, WhisperAsr
import os
from pathlib import Path
from datetime import datetime
import csv


class PipeLine:
    def __init__(self, models: list['AsrModel'], categories: list[str]):
        self.models = models
        self.dataset_paths = categories
        self.output_file_path = Path('ASR output')

    def run(self, normalize=False):

        headers = ['name', 'category', 'model', 'time', 'transcript']
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        with open(Path('ASR output') / Path(f"{current_time}.csv"), 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()

            for model in self.models:
                model = model()
                print(f" x-x-x-x-x INITIALIZED MODEL {model.name} x-x-x-x-x ")
                for dataset_path in self.dataset_paths:
                    print(f" x-x-x-x-x loading data for {ds.name} x-x-x-x-x ")
                    ds = DS(dataset_path)
                    transcripts, times = model.transcribe(ds)

                    for name in transcripts.keys():
                        writer.writerow({
                            'name': name,
                            'category': dataset_path,
                            'transcript': transcripts[name],
                            'model': model.name,
                            'time': times[name]
                        })


def format_output():
    path = Path("ASR output")


a = PipeLine(models=[WhisperAsr, ParakeetAsr, CohereAsr, CanaryAsr],
             categories=['Test'])
a.run()
