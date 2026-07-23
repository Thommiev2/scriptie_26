import csv
import os
from pathlib import Path

from utility_functions import count_words, ped_ges, normalize
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
        logs = open('logs.txt', 'w')
        # Every GenAI model in 'summaries'
        for genai_model in os.listdir(Path('summaries')):
            print(f"[SYS]    Generating scores for {genai_model[genai_model.find('_') + 1:]}")

            base_header = ['name', 'category', 'asr_model', 'genai_model', 'sum_num_words']
            with open(Path('output/summarization scoring') / f"{genai_model[genai_model.find('_') + 1:]}.csv", 'w', newline='', encoding='utf-8') as f_w:
                writer = csv.DictWriter(f_w, fieldnames=base_header + [metric.name for metric in self.accuracy_metrics] + [metric.name for metric in self.metrics])
                writer.writeheader()
                # Every summary inside created by the GenAI model. also does gt on gt for verification.
                for summary in os.listdir(Path('summaries') / genai_model):
                    try:
                        with open(Path('summaries') / genai_model / summary, 'r', encoding='utf-8') as f_r:
                            # remove special tokens
                            f_r_text = f_r.read()

                            print(f"[SYS]    Processing {summary}")

                            model, category, name = summary[:summary.rfind('.')].split('_')
                            if name == 'ge':
                                continue


                            # # Set up lossless summary data structure and base row
                            # reference_summaries = {}
                            # for name in os.listdir(Path('summaries')):
                            #     if row['asr_model'] == 'gt':
                            #         reference_summaries[f"{row['name']}_{row['category']}"] = row['summary']

                            # print(reference_summaries.keys())
                            #
                            # f_r.seek(0)
                            # reader = csv.DictReader(f_r)
    #
                            # for row in reader:
                            #     reference_summary = reference_summaries[f"{row['name']}_{row['category']}"]
                            #     transcript = clean_func[row['category']](open(Path('../dataset') / row['category'] / Path(f"ground truth/{row['name']}.txt")).read())

                            output_row = {
                                'name': name,
                                'category': category,
                                'asr_model': model,
                                'genai_model': genai_model,
                                'sum_num_words': count_words(f_r_text)
                            }
    #
                            # Write the accuracy score to the output row and get the lossless summary
                            with open(Path('summaries') / genai_model / f"gt_{category}_{name}.md", 'r', encoding='utf-8') as f_reference:
                                f_reference_text = f_reference.read()
                                for metric in self.accuracy_metrics:
                                    print(f"[SYS]    Calculating {metric.name}")
                                    output_row[metric.name] = metric.calculate_score(f_reference_text, f_r_text)
                                f_reference.close()

                            # Write the others
                            with open(Path('dataset') / category / Path('ground truth') / f"{name}.txt", 'r', encoding='utf-8') as f_transcript:
                                f_transcript_text = ped_ges(f_transcript.read())
                                for metric in self.metrics:
                                    print(f"[SYS]    Calculating {metric.name}")
                                    output_row[metric.name] = metric.calculate_score(f_transcript_text, f_r_text)
                                f_transcript.close()
                            f_r.close()
                        writer.writerow(output_row)

                    except:
                        print(f"[SYS] X  x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
                        logs.write(f"         Error while computing {summary} by {genai_model}")
                        print(f"         x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x-x")
                f_w.close()

if __name__ == '__main__':
    a = PipeLine4([BertScore, ROUGE], [CompressionRate, Density, Coverage, SummaC])
    a.run()

