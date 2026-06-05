import csv
import os
from utility_functions import dok_pat, ped_ges
from genai_models import GPT, Gemini, GenAI
from pathlib import Path


#
#
#       This file runs every generativeAI model with a prompt and instructions on every file in ASR output
#       as well as on the ground truths
#       It outputs a csv file is GenAI output for each file and is structured as follows:
#
#       | GenAI output
#       | - [ASR output file name].csv  ->  name, category, asr_model, genai_model, summary
#
#       asr_models: [asr models] | ground_truth
#
#   name,category,model,time,transcript


clean_func = {'Dokter Patient': dok_pat, 'Pedagogische gesprekken': ped_ges, 'Test': dok_pat}


class PipeLine3:
    def __init__(self, models: list['GenAI']):
        self.models = models

    def run(self):

        headers = ['name', 'category', 'asr_model', 'genai_model', 'summary']

        for file in os.listdir(Path('ASR output')):
            if file in os.listdir(Path('GenAI output')):
                print(f"ASR OUTPUT FILE {file} HAS ALREADY BEEN SUMMARIZED")
                continue
            with open(Path('ASR output') / file, 'r', encoding='utf-8') as f_r:
                reader = csv.DictReader(f_r)
                with open(Path('GenAI output') / file, 'w', newline='', encoding='utf-8') as f_w:
                    writer = csv.DictWriter(f_w, fieldnames=headers)
                    writer.writeheader()

                    for model in self.models:
                        model = model()
                        print(f" x-x-x-x-x INITIALIZED MODEL {model.model} x-x-x-x-x ")
                        for row in reader:
                            summary = model.summarize_single_call(text=row['transcript'],
                                                                  name=row['name'],
                                                                  category=row['category'],
                                                                  model=row['model'])
                            writer.writerow({
                                'name': row['name'],
                                'category': row['category'],
                                'asr_model': row['model'],
                                'genai_model': model.model,
                                'summary': summary
                            })

                    for category in os.listdir(Path('dataset')):
                        gt_path = Path('dataset') / category / Path('ground truth')
                        for transcript in os.listdir(gt_path):
                            summary = model.summarize_single_call(text=clean_func[category](open(gt_path / transcript).read()),
                                                                  name=row['name'],
                                                                  category=row['category'],
                                                                  model='gt')

                            writer.writerow({
                                'name': transcript,
                                'category': category,
                                'asr_model': 'gt',
                                'genai_model': model.model,
                                'summary': summary
                            })
                f_w.close()
            f_r.close()
            print(f' x-x-x-x-x SUCCESFULLY GENERATED SUMMARIES FOR ASR OUTPUT FILE {file} x-x-x-x-x ')

a = PipeLine3(models=[Gemini, GPT])
a.run()


