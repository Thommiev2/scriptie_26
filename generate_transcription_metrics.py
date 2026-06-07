from metrics import WER, SimDist
from utility_functions import clean_func, count_words, count_sentences
import os
from pathlib import Path
import csv

#
#
#       This file calculates every transcription metric for every ASR generated transcript
#       (Hypothesis) with the human made transcript (Ground Truth) as reference.
#       It outputs a single csv file with all the scores as follows
#
#       | output
#       | - transcript scoring
#         | - year-month-day_hour-min-second.csv  ->  name, category, model, num_words, metrics ...
#
#


class PipeLine2:
    def __init__(self, metrics):
        self.asr_output = Path("output/asr output")
        self.metrics = [metric() for metric in metrics]

    def run(self):

        headers = ['name', 'category', 'model', 'num_words', 'num_sentences'] + [metric.name for metric in self.metrics]
        already_calculated = os.listdir(Path('output/transcript scoring'))
        print(already_calculated)

        for file in os.listdir(self.asr_output):
            print(file)
            if file not in already_calculated:
                with open(Path('output/transcript scoring') / file, 'w', newline='', encoding='utf-8') as f_w:

                    writer = csv.DictWriter(f_w, fieldnames=headers)
                    # print(writer.fieldnames)
                    writer.writeheader()
                    with open(self.asr_output / file, 'r', encoding='utf-8') as f_r:
                        reader = csv.DictReader(f_r)
                        for row in reader:

                            ground_truth_path = Path('dataset') / row['category'] / Path('ground truth') / Path(f"{row['name']}.txt")
                            ground_truth = open(ground_truth_path).read()
                            ground_truth = clean_func[row['category']](ground_truth).strip()
                            hypothesis = row['transcript'].strip()
                            #
                            # print(ground_truth)
                            # print(hypothesis)

                            # simd_score, num_sentences = simd.calculate_score(ground_truth, hypothesis)
                            # wer_score, num_words = wer.calculate_score(ground_truth, hypothesis)
                            # print(row, type(row))

                            output_row = {
                                'name': row['name'],
                                'category': row['category'],
                                'model': row['model'],
                                'num_words': count_words(ground_truth),
                                'num_sentences': count_sentences(ground_truth),
                            }

                            for metric in metrics:
                                output_row[metric.name] = metric.calculate_score(ground_truth, hypothesis)

                            writer.writerow(output_row)

                        f_r.close()
                    f_w.close()

a = PipeLine2()
a.run()


