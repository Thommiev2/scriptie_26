from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from data_cleaning import align_sentences, normalize

import numpy as np
import torch
import jiwer


class Metric:
    def __init__(self):
        self.normalizer = normalize

    def normalize(self, text):
        self.normalizer(text)


class SummaC:
    def __init__(self):
        self.model = AutoModelForSequenceClassification.from_pretrained("LokaalHub/nl-nli-klein")
        self.tokenizer = AutoTokenizer.from_pretrained("LokaalHub/nl-nli-klein")

    def calculate_score(self, gt, h) -> float:

        if isinstance(gt, str):
            gt = gt.split('.')
        if isinstance(h, str):
            h = h.split('.')

        # Array containing entailment scores for each pair of summary sentences with sentences from the original document
        array = np.zeros((len(gt), len(h)))

        for i, gt_sentence in enumerate(gt):
            for j, h_sentence in enumerate(h):

                inputs = self.tokenizer(gt_sentence, h_sentence, truncation='only_first', max_length=256, return_tensors='pt')

                with torch.no_grad():
                    logits = self.model(**inputs).logits

                prob_distr = logits.softmax(-1).squeeze().tolist()
                # Takes just the entailment probability. Change later to incorporate other scores into single value
                array[i, j] = prob_distr[0]
                print(f"gt: {gt_sentence}, h: {h_sentence}, entailment_score: {prob_distr[0]}")

        max_entailment_scores = np.max(array, axis=0)

        return np.average(max_entailment_scores)


class WER:
    def __init__(self):
        pass

    def calculate_score(self, gt, h) -> float:

        if not isinstance(gt, str):
            raise TypeError(f"Ground Truth is not of type string but of type {type(gt)} while calculating WER")
        if not isinstance(h, str):
            raise TypeError(f"Hypothesis is not of type string but of type {type(h)} while calculating WER")

        processed_words = jiwer.process_words(gt, h)
        return processed_words.wer


class SimDist:
    def __init__(self):
        self.model = SentenceTransformer('NetherlandsForensicInstitute/robbert-2022-dutch-sentence-transformers')

    def calculate_score(self, gt, h) -> float:

        gt = self.model.encode(gt)
        h = self.model.encode(h)

        sim_values = np.zeros(len(gt))

        for i, (gt_sent, h_sent) in enumerate(zip(gt, h)):

            # Normalize vectors
            gt_sent = gt_sent / np.linalg.norm(gt_sent)
            h_sent = h_sent / np.linalg.norm(h_sent)

            # Calculate angle between 0 to 1
            sim_values[i] = np.dot(gt_sent, h_sent)

        # average semantic difference
        return np.sum(sim_values, dtype=float) / len(gt)


class CompressionRate:
    def __init__(self):
        pass

    def calculate_score(self, gt, h):

        if not isinstance(gt, str):
            raise TypeError(f"Ground Truth is not of type string but of type {type(gt)} while calculating Compression Rate")
        if not isinstance(h, str):
            raise TypeError(f"Hypothesis is not of type string but of type {type(h)} while calculating Compression Rate")

        return len(h) / len(gt)


class Density:
    def __init__(self):
        pass

    def calculate_score(self, gt, h):
        sequences = get_common_sequences(gt, h)
        score = sum(len(seq)**2 for seq in sequences['sequences']) / sequences['sum_length']
        return score


class Coverage:
    def __init__(self):
        pass

    def calculate_score(self, gt, h):
        sequences = get_common_sequences(gt, h)
        score = sum(len(seq) for seq in sequences['sequences']) / sequences['sum_length']
        return score


class ROUGE:
    def __init__(self, n):
        self.seq_length = n

    def calculate_score(self, gt, h):

        ref_sum = gt.split(' ')
        summ = h.split(' ')

        sum_index = 0
        ref_sum_index = 0
        match_count = 0

        while ref_sum_index <= len(ref_sum) - self.seq_length:
            while sum_index <= len(summ) - self.seq_length:
                n = 0
                while n < self.seq_length:
                    if ref_sum[ref_sum_index + n] == summ[sum_index + n]:
                        n += 1
                    else:
                        print(ref_sum[ref_sum_index: ref_sum_index + n])
                        break
                else:
                    match_count += 1
                    print(ref_sum[ref_sum_index: ref_sum_index + n])
                    break
                sum_index += 1
            sum_index = 0
            ref_sum_index += 1

        return match_count / (len(ref_sum) - self.seq_length + 1)


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
# gt2 = "hallo meneer ik ben thomas"
# h2 = "hallo man ik ben thomas"
# h3 = "hallo hond ik ben thomas"
#
# w = SimDist()
# a = w.calculate_score(gt2, h2)
# b = w.calculate_score(gt2, h3)
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

# #################### ROUGE #################################
# a = "hello my name stoer is thomas"
# b = "hello my name cool is thomas"
#
# d = ROUGE()
# print(d.calculate_score(2, a, b))
