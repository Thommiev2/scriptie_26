from metrics import WER, SimDist
from utility_functions import dok_pat, ped_ges
import os
from pathlib import Path
import csv

#
#
#       This file runs every ASR metric on every ASR generated transcript
#       (Hypothesis) on the human made transcript (Ground Truth).
#       It outputs a single csv file with all the scores as follows
#
#       | Metrics
#       | - year-month-day_hour-min-second.csv  ->  name, category, model, wer, simdist, num_words
#
#

clean_func = {'Dokter Patient': dok_pat, 'Pedagogische gesprekken': ped_ges, 'Test': dok_pat}


class PipeLine2:
    def __init__(self):
        self.asr_output = Path("ASR output")

    def run(self):

        headers = ['name', 'category', 'model', 'wer', 'simdist', 'num_words', 'num_sentences']
        already_calculated = os.listdir(Path('Scoring transcript'))
        print(already_calculated)

        for file in os.listdir(self.asr_output):
            print(file)
            if file not in already_calculated:
                with open(Path('Scoring transcript') / file, 'w', newline='', encoding='utf-8') as f_w:

                    writer = csv.DictWriter(f_w, fieldnames=headers)
                    # print(writer.fieldnames)
                    writer.writeheader()
                    with open(self.asr_output / file, 'r', encoding='utf-8') as f_r:
                        reader = csv.DictReader(f_r)
                        simd, wer = SimDist(), WER()
                        # print(reader.fieldnames)
                        for row in reader:

                            ground_truth_path = Path('dataset') / row['category'] / Path('ground truth') / Path(f"{row['name']} transcript.txt")
                            ground_truth = open(ground_truth_path).read()
                            ground_truth = clean_func[row['category']](ground_truth).strip()
                            hypothesis = row['transcript'].strip()

                            print(ground_truth)
                            print(hypothesis)

                            simd_score, num_sentences = simd.calculate_score(ground_truth, hypothesis)
                            wer_score, num_words = wer.calculate_score(ground_truth, hypothesis)
                            # print(row, type(row))

                            writer.writerow({
                                'name': row['name'],
                                'category': row['category'],
                                'model': row['model'],
                                'wer': wer_score,
                                'simdist': simd_score,
                                'num_words': num_words,
                                'num_sentences': num_sentences,
                            })

                        f_r.close()
                    f_w.close()

a = PipeLine2()
a.run()


