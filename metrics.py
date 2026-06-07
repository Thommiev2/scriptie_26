import re

from transformers import AutoTokenizer, AutoModelForSequenceClassification, RobertaTokenizer
from sentence_transformers import SentenceTransformer
from utility_functions import align_sentences, normalize

import numpy as np
import torch
import jiwer
from bert_score import score
from pathlib import Path

class SummaC:
    def __init__(self):
        self.name = 'summac'
        self.model = AutoModelForSequenceClassification.from_pretrained("LokaalHub/nl-nli-klein")
        self.tokenizer = AutoTokenizer.from_pretrained("LokaalHub/nl-nli-klein")

    def calculate_score(self, transcript: str, summary: str) -> (float, int):

        # validate input
        if not isinstance(transcript, str):
            raise TypeError(f"Transcript is not of type string but of type {type(transcript)}")
        if not isinstance(summary, str):
            raise TypeError(f"Summary is not of type string but of type {type(summary)}")

        transcript = re.sub(r"[?!]+", '.', transcript)
        h = re.sub(r"[?!]+", '.', summary)

        transcript = transcript.split('.')
        summary = h.split('.')

        # Array containing entailment scores for each pair of summary sentences with sentences from the original document
        array = np.zeros((len(transcript), len(summary)))

        for i, transcript_sentence in enumerate(transcript):
            for j, summary_sentence in enumerate(summary):

                inputs = self.tokenizer(transcript_sentence, summary_sentence, truncation='only_first', max_length=256, return_tensors='pt')

                with torch.no_grad():
                    logits = self.model(**inputs).logits

                prob_distr = logits.softmax(-1).squeeze().tolist()
                # Takes just the entailment probability. Change later to incorporate other scores into single value
                array[i, j] = prob_distr[0]
                # print(f"gt: {transcript_sentence}, h: {summary_sentence}, entailment_score: {prob_distr[0]}")

        max_entailment_scores = np.max(array, axis=0)

        return np.average(max_entailment_scores)


class WER:
    def __init__(self):
        self.name = 'wer'

    def calculate_score(self, gt, h) -> (float, int):

        if not isinstance(gt, str):
            raise TypeError(f"Ground Truth is not of type string but of type {type(gt)} while calculating WER")
        if not isinstance(h, str):
            raise TypeError(f"Hypothesis is not of type string but of type {type(h)} while calculating WER")

        gt = normalize(gt)
        h = normalize(h)

        processed_words = jiwer.process_words(gt, h)
        return processed_words.wer


class SimDist:
    def __init__(self):
        self.name = 'simdist'
        self.model = SentenceTransformer('NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers')

    def calculate_score(self, gt: str, h: str) -> (float, int):

        gt, h = align_sentences(gt, h)

        gt = [normalize(sentence) for sentence in gt]
        h = [normalize(sentence) for sentence in h]

        gt_embed = self.model.encode(gt, convert_to_numpy=True)
        h_embed = self.model.encode(h, convert_to_numpy=True)
        # print(gt_embed)

        gt_norm = gt_embed / np.linalg.norm(gt_embed, axis=1, keepdims=True)
        h_norm = h_embed / np.linalg.norm(h_embed, axis=1, keepdims=True)

        sim_values = gt_norm * h_norm
        sim_values = np.sum(sim_values, axis=1)
        # print(sim_values)

        return float(np.mean(sim_values))


class CompressionRate:
    def __init__(self):
        self.name = 'compression_rate'

    def calculate_score(self, transcript: str, summary: str):

        if not isinstance(transcript, str):
            raise TypeError(f"Ground Truth is not of type string but of type {type(transcript)} while calculating Compression Rate")
        if not isinstance(summary, str):
            raise TypeError(f"Hypothesis is not of type string but of type {type(summary)} while calculating Compression Rate")

        return len(summary) / len(transcript)


class BertScore:
    def __init__(self):
        # self.model = SentenceTransformer('NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers')
        self.name = 'bertscore'

    def calculate_score(self, reference: str, candidate: str):

        # ref_tokens = self.tokenizer.tokenize(reference)
        # can_tokens = self.tokenizer.tokenize(candidate)
        #
        # for token in ref_tokens:
        #     print(token)
        # for token in can_tokens:
        #     print(token)

        reference = [reference]
        candidate = [candidate]

        if not Path.exists(Path("hf_model")):
            model = SentenceTransformer('NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers')
            model[0].auto_model.save_pretrained("./hf_model")
            model.tokenizer.save_pretrained("./hf_model")

        precission_score, recall_score, f1_score = score(
            candidate,
            reference,
            model_type="./hf_model",
            num_layers=12,

        )

        return [precission_score.item(), recall_score.item(), f1_score.item()]


class Density:
    def __init__(self):
        self.name = 'density'

    def calculate_score(self, transcript: str, summary: str):
        sequences = get_common_sequences(transcript, summary)
        score = sum(len(seq)**2 for seq in sequences['sequences']) / sequences['sum_length']
        return score


class Coverage:
    def __init__(self):
        self.name = 'coverage'

    def calculate_score(self, transcript: str, summary: str):
        sequences = get_common_sequences(transcript, summary)
        score = sum(len(seq) for seq in sequences['sequences']) / sequences['sum_length']
        return score


class ROUGE:
    def __init__(self, n=2):
        self.name = 'rouge'
        self.seq_length = n

    def calculate_score(self, reference: str, candidate: str):

        ref_summ = reference.split(' ')
        summ = candidate.split(' ')

        sum_index = 0
        ref_summ_index = 0
        match_count = 0

        while ref_summ_index <= len(ref_summ) - self.seq_length:
            while sum_index <= len(summ) - self.seq_length:
                n = 0
                while n < self.seq_length:
                    if ref_summ[ref_summ_index + n] == summ[sum_index + n]:
                        n += 1
                    else:
                        # print(ref_summ[ref_summ_index: ref_summ_index + n])
                        break
                else:
                    match_count += 1
                    # print(ref_summ[ref_summ_index: ref_summ_index + n])
                    break
                sum_index += 1
            sum_index = 0
            ref_summ_index += 1

        return match_count / (len(ref_summ) - self.seq_length + 1)


def get_common_sequences(original_text, summary):

    if isinstance(original_text, str):
        original_text = original_text.split(' ')
    if isinstance(summary, str):
        summary = summary.split(' ')

    l_sum = len(summary)
    l_text = len(original_text)

    sequences: set = set()
    sum_index, text_index = 0, 0
    while sum_index < l_sum:
        seq = []
        while text_index < l_text:
            if summary[sum_index] == original_text[text_index]:
                # print(summary[sum_index], original_text[text_index])
                sum_seq_index = sum_index
                text_seq_index = text_index
                while summary[sum_seq_index] == original_text[text_seq_index]:
                    sum_seq_index += 1
                    text_seq_index += 1
                    if text_seq_index == l_text or sum_seq_index == l_sum:
                        break
                # print(len(seq), (sum_seq_index - sum_index))
                if len(seq) < (sum_seq_index - sum_index):
                    seq = summary[sum_index:sum_seq_index]
                text_index = text_seq_index
            else:
                text_index += 1
        sum_index = sum_index + max(len(seq), 1)
        text_index = 0
        sequences.add(tuple(seq))

    return {'sequences': sequences, 'sum_length': l_sum}

########################################### SummaC ################################################
# premise = "Het contract gaat in op 1 mei 2026. Het is koud. Ik ben cool. oops"
# hypothesis = "Het contract start in mei. vandaag voelt het koud aan"
#
# s = SummaC()
# print(s.calculate_score(premise, hypothesis))

########################################### WER ################################################
# GT = """
#     Code-switching: Het vloeiend overstappen van de ene taal naar de andere binnen een zin of gesprek.
#     Bijvoorbeeld: "De patiënt heeft een goede prognosis, maar we moeten wel de follow-up in de gaten houden."
#     """
#
# H = """
#     Code-switching we: Het overstappen van de ene taal naar de binnen een zin of gesprek hi.
#     Bijvoorbeeld: "De ptiënt hft een goede prognosis, maar we moeten wel de follow-up in de gaten houden."
#     """
#
# GT1 = "de kat loopt.\n hallo meneer. "
# H1 = "de loopt.\n hallo meneer sam"
#
# w = WER()
# w.calculate_score(GT1, H1)

######################################## SimDist ###########################################
# GT = """
#     Code-switching: Het vloeiend overstappen van de ene taal naar de andere binnen een zin of gesprek.
#     Bijvoorbeeld: "De patiënt heeft een goede prognosis, maar we moeten wel de follow-up in de gaten houden."
#     """
#
# H = """
#     Code-switching we: Het overstappen van de ene taal naar de binnen een zin of gesprek hi.
#     Bijvoorbeeld: "De ptiënt hft een goede prognosis, maar we moeten wel de follow-up in de gaten houden."
#     """
#
# gt2 = ["hallo meneer ik ben thomas", "hallo meneer ik ben thomas", "hallo meneer ik ben thomas", "hallo meneer ik ben thomas"]
# h2 = ["hallo man ik ben thomas", "hallo mevrouw ik ben thomas", "hallo hond ik ben thomas", "shuuush ff niet praten"]
# # h3 = "hallo hond ik ben thomas"
# #
# w = SimDist()
# a = w.calculate_score(gt2, h2)
# b = w.calculate_score(gt2, gt2)
# print(a, b)
# c = w.calculate_score(h3, h2)
# print(a, b, c)

# ####################### Compression Rate ##################################
# gt = "hallo iedereen. hjoe het gaat met jullie, ik ben thomas en ik ga iets vertellen iver uets"
# h = "thomas gaat iets vertellen"
#
# a = CompressionRate()
# print(a.calculate_score(gt, h))

########################### sequences #######################################
# a = """"
# Following identification and de-duplication, we
# extracted the article texts and summaries and further cleaned and filtered the dataset.
# Article Text We used Readability5
# to extract
# HTML body content. Readability uses HTML
# heuristics to extract the main content and title of
# a page, producing article text without extraneous
# HTML markup and images. Our preliminary testing, as well as comparison by Peters (2015), found
# Readability to be one of the highest accuracy content extraction algorithms available. To exclude
# inline advertising and image captions sometimes
# present in extractions, we applied additional filtering of paragraphs with fewer than five words. We
# excluded articles with no body text extracted.
# Summary Metadata We extracted the article
# summaries from the metadata available in the
# HTML pages of articles. These summaries are
# often written by newsroom editors and journalists to appear in social media distribution and
# search results. While there is no standard metadata format for summaries online, common fields
# are often present in the page’s HTML. Popular
# metadata field types include: og:description, twitter:description, and description. In cases where"""
#
# b = "extracted the hello image and no body text appear extracted twitter field types can not believe through accuracy body content highest excluded insane new story metadata"
#
# gt = "yam nam happy happy yam yam"
# h = "nam happy yam nam nam happy"
#
# gt1 = "hi ha ho"
# h1 = "l hi l"
#
# aa = Coverage()
# bb = Density()
# print(get_common_sequences(a, b))
# print(aa.calculate_score(a, b))
# print(bb.calculate_score(a, b))
# print(get_common_sequences(gt1, h1))
# print(aa.calculate_score(gt1, h1))
# print(bb.calculate_score(gt1, h1))

#################### ROUGE #################################
# a = "hello my name stoer is thomas. ik hou van zingen en spelletje"
# b = "hello my name cool is thomas. spelletjes en zingen vind ik leuk, maar niet heus"
#
# d = BertScore()
# print(d.calculate_score(a, b))
