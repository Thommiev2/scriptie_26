import re
import jiwer
from whisper.normalizers import BasicTextNormalizer

normalizer = BasicTextNormalizer(remove_diacritics=True)
basic_filler_words_regex = re.compile(r"\s[d]?[uea]+[hm]+\b", flags=re.MULTILINE)


rules = {
    'het': ["t"],
    'goededag': ["goeiedag", "goede dag", "goeie dag"],
    'mijn': ["me"],
    'het is': ["tis"],
    'oke': ["ok"]
}


def normalize(text):
    text = normalizer(text)
    text = basic_filler_words_regex.sub("", text)
    return text


def delete_unaudible_segments(gt, h):

    aligned_text = jiwer.process_words(gt, h)

    for index, word in enumerate(gt):
        if word == '[unaudible]':
            i = 0
            while aligned_text.alignments[index].ref:
                i += 1

def align_sentences(gt, h):

    aligned_text = jiwer.process_words(gt, h)

    reference = []
    hypothesis = []

    alignment_index = 0
    last_index = 0
    last_index_h = 0

    for index, word in enumerate(aligned_text.references[0]):
        print("index", index)
        if word[-1] == '.':
            reference.append(' '.join(aligned_text.references[0][last_index:index + 1]))
            offset = index - (last_index - last_index_h)
            print(last_index, last_index_h)
            while True:
                print(alignment_index)
                chunk = aligned_text.alignments[0][alignment_index]
                if chunk.type == 'delete' and last_index_h != chunk.hyp_end_idx:
                    offset -= chunk.ref_end_idx - chunk.ref_start_idx
                if chunk.type == 'insert' and last_index != chunk.ref_end_idx:
                    offset += chunk.hyp_end_idx - chunk.hyp_start_idx

                if chunk.ref_start_idx < index < chunk.ref_end_idx or index == chunk.ref_start_idx:
                    hypothesis.append(' '.join(aligned_text.hypotheses[0][last_index_h:offset + 1]))
                    break
                alignment_index += 1

            last_index = index + 1
            last_index_h = offset + 1

    return reference, hypothesis


t = "hallo ik ben. door de woorden. wow crazy aaa cool, nice. asdw aaawdw. asdwaf wad adwa awwww. maar dan. huh hoeze. crazy work haha. gekke handoek. maar dan niet echt."
s = "hallo ik door de woorden. wow aaa crazy, nice. asdw aaawdw. asdwaf wad adwa. awwww hoi. maar dan. huh hoezo no way haha. gekke lol nee handoek cool maar dan wel"
p = "ik [unaudible] zeg"
r = "ik doe als jij zeg"
error = jiwer.process_words(p, r)
print(error)
# # print(error.alignments[0][0].ref_start_idx)
# print(error.alignments[0][0])
# print(align_sentences(error))

# t = """Leger des heils, Christian health care organization,
# van uit  christelijke waarde helpen uhhh ummm duuh ahhhh zë mensën die dat zelf niet kunnen en/of nodig hebben
# most know for  providing shelter to the homeless
# many different    branches like child protective services"""
# h = 'uhh ahhh ammmm hallo sub'
# print(jiwer.process_words(t,h))
# print(normalize(t))
