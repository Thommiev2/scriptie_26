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
#       - output
#       | - genai output
#         | - [asr output file name].csv  ->  name, category, asr_model, genai_model, summary
#
#       asr_models: [asr models] | ground_truth
#
#   name,category,model,time,transcript


clean_func = {'Dokter Patient': dok_pat, 'Pedagogische gesprekken': ped_ges, 'Test': dok_pat}


class PipeLine3:
    def __init__(self, models: list['GenAI']):
        self.models = [model() for model in models]

    def run(self):

        headers = ['name', 'category', 'asr_model', 'genai_model', 'summary']

        for file in os.listdir(Path('output/asr output')):
            if file in os.listdir(Path('output/genai output')):
                print(f"X ASR OUTPUT FILE {file} HAS ALREADY BEEN SUMMARIZED")
                continue
            with open(Path('output/asr output') / file, 'r', encoding='utf-8') as f_r:
                reader = csv.DictReader(f_r)
                with open(Path('output/genai output') / file, 'w', newline='', encoding='utf-8') as f_w:
                    writer = csv.DictWriter(f_w, fieldnames=headers)
                    writer.writeheader()

                    unique_names = set([])
                    unique_categories = set([])

                    for row in reader:
                        for model in self.models:
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

                            unique_names.add(row['name'])
                            unique_categories.add(row['category'])

                    for category in unique_categories:
                        gt_path = Path('dataset') / category / Path('ground truth')
                        for name in unique_names:
                            summary = model.summarize_single_call(text=clean_func[category](open(gt_path / Path(f'{name}.txt')).read()),
                                                                  name=name,
                                                                  category=category,
                                                                  model='gt')

                            writer.writerow({
                                'name': name,
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


