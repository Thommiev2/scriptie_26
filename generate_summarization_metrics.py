import csv
import os
from pathlib import Path

from utility_functions import count_words, clean_func
from metrics import BertScore, ROUGE, SummaC, Coverage, CompressionRate, Density


#
#
#       This file calculate every summarization metric for every GenAI generated summary
#       (summary/candidate) with the original transcript / loss less summary (transcript/reference).
#       It outputs a single csv file containing both the summarization accuracy metrics (ROUGE, BERTScore)
#       and all the other (SummaC, compression rate, coverage, density)
#
#       - output
#       | - summarization scoring
#         | - [GenAI output filename].csv -> name, category, asr_model, genai_model, metrics ...
#
#


class PipeLine4:
    def __init__(self, accuracy_metrics, metrics):
        self.accuracy_metrics = [metric() for metric in accuracy_metrics]
        self.metrics = [metric() for metric in metrics]

    def run(self):
        for genai_file in os.listdir(Path('output/genai output')):
            if genai_file in [sum_score_file for sum_score_file in os.listdir(Path('output/summarization scoring'))]:
                print(f"X GENAI OUTPUT FILE {genai_file} HAS ALREADY BEEN SUMMARIZED")
                continue
            with open(Path('output/genai output') / genai_file, 'r', encoding='utf-8') as f_r:

                base_header = ['name', 'category', 'asr_model', 'genai_model', 'sum_num_words']
                with open(Path('output/summarization scoring') / genai_file, 'w', newline='', encoding='utf-8') as f_w:
                    writer = csv.DictWriter(f_w, fieldnames=base_header + [metric.name for metric in self.accuracy_metrics] + [metric.name for metric in self.metrics])
                    writer.writeheader()

                    # Set up lossless summary data structure and base row
                    reader = csv.DictReader(f_r)
                    reference_summaries = {}
                    for row in reader:
                        if row['asr_model'] == 'gt':
                            reference_summaries[f"{row['name']}_{row['category']}"] = row['summary']

                    print(reference_summaries.keys())

                    f_r.seek(0)
                    reader = csv.DictReader(f_r)

                    for row in reader:
                        reference_summary = reference_summaries[f"{row['name']}_{row['category']}"]
                        transcript = clean_func[row['category']](open(Path('dataset') / row['category'] / Path(f"ground truth/{row['name']}.txt")).read())

                        output_row = {
                            'name': row['name'],
                            'category': row['category'],
                            'asr_model': row['asr_model'],
                            'genai_model': row['genai_model'],
                            'sum_num_words': count_words(reference_summary)
                        }

                        # Write the accuracy score file
                        for metric in self.accuracy_metrics:
                            output_row[metric.name] = metric.calculate_score(reference_summary, row['summary']) if row['asr_model'] != 'gt' else ""

                        # Write the others
                        for metric in self.metrics:
                            output_row[metric.name] = metric.calculate_score(transcript, row['summary'])

                        writer.writerow(output_row)

                    f_w.close()
                f_r.close()


a = PipeLine4([BertScore, ROUGE], [CompressionRate, Density, Coverage, SummaC])
a.run()
