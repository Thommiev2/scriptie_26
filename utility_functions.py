import csv
import pathlib

import regex as re
import jiwer
from whisper.normalizers import BasicTextNormalizer

# CLEAN TRANSCRIPT AND GROUND TRUTH BEFORE SCORE CALCULATION


# initialize OpenAI's basic text normalizer, removes "  ", special characters, punctuation, capitalization
normalizer = BasicTextNormalizer(remove_diacritics=True)
# basic regex expression for filler words, matches duhh, uhhh, hmmm, emm, etc
basic_filler_words_regex = re.compile(
    r"(?:^|\s)[d]?[uea]*[hm]+(?=\P{L}|$)",
    flags=re.MULTILINE | re.IGNORECASE,
)
# basic regex expression for o and ohhh
other_filler_words_regex = re.compile(
    r"(?:^|\s)[o]+[h]*(?=\P{L}|$)",
    flags=re.MULTILINE | re.IGNORECASE,
)

# rule-based lexicon, matches words with similar meaning (not functional yet, can be expanded on during transcribing ground truth)
lexicon: dict[str, list[str]] = {
    'het': ["t", "'t"],
    'goede': ['goeie'],
    'goededag': ["goeiedag", "goede dag", "goeie dag"],
    'mijn': ["mn", "m'n"],
    'het is': ["tis", "ts", "t's", "'ts"],
    'oke': ["ok", 'k'],
    'er': ["d'r"],
    'snachts': ["'s nachts", "s nachts"],
    'savonds': ["s avonds", "'s avonds"],
    'sochtends': ["'s ochtends", 's ochtends'],
    'smiddags': ["'s middags", 's middags'],
    'zo een': ["zo'n"],
    'nou ja': ["nouja"],
    '': [' hè', ' hé'],
    'niet': ['nie', 'ni'],
    'elkaar': ['mekaar'],
    'etcetera': ['et cetera', 'etc'],
    'procent': ['%'],
    'euro': ['€'],
    'eigenlijk': ['eigelijk'],
    'en zo': ['enzo'],
    'wauw': ['wow', 'wouw'],
}


def normalize(text: str, level: int=3):

    """
    Applies all the normalization functions outlined above.
    NOTE: Always keep on level 3. Apply to both the GT and H before calculating ASR scores.

    :param text:    (str) Hypothesis or ground truth text file stored as a string
    :param level:   (int) The level of abstraction, higher level means more normalization functions applied.
        level 1: Basic lower casing and removal of punctuation
        level 2: Apply basic lexicon and removal of filler words
        level 3: Remove all stutter sequences
    :return:        (str) normalized text
    """

    if level > 0:
        text = text.lower()
    if level > 1:
        text = basic_filler_words_regex.sub("", text)
        text = other_filler_words_regex.sub("", text)
        text = apply_ruleset(text, lexicon)
    if level > 0:
        text = normalizer(text)
    if level > 2:
        text = remove_stutters(text)

    return text


def remove_stutters(text, sequence_length=1) -> str:

    """
    Removes sequences of the same word.
    NOTE: Can be set to a higher length but I didn't encounter any stutter sequence with a length of >3.
    Setting the length higher increases false positives.

    :param text:                (str) Hypothesis or ground truth text file
    :param sequence_length:     (int) sequence length of repeating words to be removed
    :return:
    """

    def remove_word_punctuation(w) -> (str, bool):
        if w == '':
            return w, False
        return w[:-1] if ((w[-1] == ',') or w[-1] == '.') else w, True if w[-1] == '.' else False

    text = text.split(' ')
    stutter_index = []

    for index in range(len(text)):
        i = sequence_length
        while index + i + sequence_length - 1 < len(text):
            match = True
            for sequence_offset in range(sequence_length):
                if remove_word_punctuation(text[index + sequence_offset])[0] != remove_word_punctuation(text[index + sequence_offset + i])[0]:
                    match = False
                    break
            if not match:
                break

            for word_index in range(sequence_length):
                stutter_index.append(index + word_index + i)
            i += sequence_length

    for stutter in sorted(set(stutter_index), reverse=True):
        text.pop(stutter)

    return ' '.join(text)


def build_pattern(variant: str) -> re.Pattern:
    """
    Helper function used in apply_ruleset()

    :param variant:     (str) the target word of the rule
    :return:            (re.Pattern) the compiled ReGeX pattern that is
    """

    escaped = re.escape(variant)
    left_needs_boundary = bool(variant) and (variant[0].isalnum() or variant[0] == "'")
    right_needs_boundary = bool(variant) and (variant[-1].isalnum() or variant[-1] == "'")

    left = r"(?<![A-Za-zÀ-ÿ0-9'])" if left_needs_boundary else ""
    right = r"(?![A-Za-zÀ-ÿ0-9'])" if right_needs_boundary else ""

    return re.compile(left + escaped + right, flags=re.IGNORECASE)


def apply_ruleset(text: str, ruleset: dict[str, list[str]] = lexicon) -> str:

    """
    Applies a ruleset to text.
    NOTE: Currently hardcoded to be the lexicon that is defined at the top.
    :param text:        (str) Hypothesis or ground truth text file stored as a string
    :param ruleset:     (dict) Rules to be applied. Consist of the target word mapped to a list of
                               words that have to be changed to the target word
    :return:            (str) text
    """

    pairs = []
    for target, variants in ruleset.items():
        for variant in variants:
            if variant == "":
                continue
            pairs.append((variant, target))
    pairs.sort(key=lambda p: len(p[0]), reverse=True)

    for variant, target in pairs:
        pattern = build_pattern(variant)
        text = pattern.sub(target, text)

    return text.strip()


def concat_csv_files(path1: pathlib.Path, path2: pathlib.Path) -> None:
    """
    Helper function that concatenates csv file outputs from different runs
    NOTE: header (field names) must be the same in both target files.

    :param path1:   (pathlib.Path) Relative or absolute path to the first csv file (this will become the new file)
    :param path2:   (pathlib.Path) Relative or absolute path to the second csv file
    :return:        None
    """
    with open(path1, 'w', newline='', encoding='utf-8') as f_w:
        writer = csv.DictWriter(f_w)
        with open(path2, 'r', encoding='utf-8') as f_r:
            reader = csv.DictReader(f_r)
            for row in reader:
                writer.writerow(row)
            f_r.close()
            path2.unlink()
        f_w.close()


def align_sentences(gt: str, h: str) -> (list[str], list[str]):
    """
    Helper function used in the calculation of SimDist. Uses a jiwer process_words object that has already
    alligned them on a word level. A chuck in this context is a sequence of words that matches (equal) or
    are part of the same error (insert, deletion, substitute).

    :param gt:              (str) ground truth transcript
    :param h:               (str) hypothesis generated by an ASR model
    :return reference:      list[str] -> the ground truth split into sentences
            hypothesis:     list[str] -> the hypothesis aligned to each ground truth sentence
    """

    gt = re.sub(r"[?!]+", '.', gt)
    h = re.sub(r"[?!]+", '.', h)

    aligned_text = jiwer.process_words(gt, h)

    ref_words = aligned_text.references[0]
    hyp_words = aligned_text.hypotheses[0]
    chunks = aligned_text.alignments[0]

    reference = []
    hypothesis = []
    ref_buffer = []
    hyp_buffer = []

    # Handles the end of sentences
    def flush():
        reference.append(' '.join(ref_buffer))
        hypothesis.append(' '.join(hyp_buffer))
        ref_buffer.clear()
        hyp_buffer.clear()

    for chunk in chunks:
        # Equal and substitution chunks are both added.
        if chunk.type in ('equal', 'substitute'):
            for ref_idx, hyp_idx in zip(
                range(chunk.ref_start_idx, chunk.ref_end_idx),
                range(chunk.hyp_start_idx, chunk.hyp_end_idx),
            ):
                ref_word = ref_words[ref_idx]
                ref_buffer.append(ref_word)
                hyp_buffer.append(hyp_words[hyp_idx])
                # '.' in the ground truth text are used to determine the end of a sentence
                if ref_word[-1] == '.':
                    flush()

        elif chunk.type == 'delete':
            for ref_idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ref_word = ref_words[ref_idx]
                ref_buffer.append(ref_word)
                if ref_word[-1] == '.':
                    flush()

        elif chunk.type == 'insert':
            hyp_buffer.extend(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])

    if ref_buffer or hyp_buffer:
        flush()

    return reference, hypothesis


# simple helper function
def count_words(text: str) -> int:
    text = normalizer(text)
    return len(text.split(' '))


# simple helper function
def count_sentences(text: str) -> int:
    text = re.sub(r"[?!]+", '.', text)
    return len(text.split('.'))


def remove_unaudible(gt: str, h: str) -> (str, str):
    """
    Helper function that removes [unaudible] parts of the ground truth and every substitution and
    insertion error that links to that word. [unaudible] was used during the ground truth creation
    for segments with ambiguous audio where the ground truth was indeterminable.

    :param gt   (str) ground truth text:
    :param h:   (str) hypothesis
    :return:    (str, str) clean gt and h
    """
    aligned_text = jiwer.process_words(gt, h)

    ref_words = aligned_text.references[0]
    hyp_words = aligned_text.hypotheses[0]
    chunks = aligned_text.alignments[0]

    ref_remove_idx = set()
    hyp_remove_idx = set()
    expand_left_chunks = set()   # chunk indices whose left-neighbor insert should be removed
    expand_right_chunks = set()  # chunk indices whose right-neighbor insert should be removed

    for chunk_i, chunk in enumerate(chunks):
        for idx in range(chunk.ref_start_idx, chunk.ref_end_idx):
            if "unaudible" not in ref_words[idx]:
                continue

            ref_remove_idx.add(idx)

            if chunk.type in ('substitute', 'equal'):
                offset = idx - chunk.ref_start_idx
                hyp_remove_idx.add(chunk.hyp_start_idx + offset)

            if idx == chunk.ref_start_idx:
                expand_left_chunks.add(chunk_i)
            if idx == chunk.ref_end_idx - 1:
                expand_right_chunks.add(chunk_i)

    for chunk_i in expand_left_chunks:
        if chunk_i - 1 >= 0 and chunks[chunk_i - 1].type == 'insert':
            prev = chunks[chunk_i - 1]
            hyp_remove_idx.update(range(prev.hyp_start_idx, prev.hyp_end_idx))

    for chunk_i in expand_right_chunks:
        if chunk_i + 1 < len(chunks) and chunks[chunk_i + 1].type == 'insert':
            nxt = chunks[chunk_i + 1]
            hyp_remove_idx.update(range(nxt.hyp_start_idx, nxt.hyp_end_idx))

    gt_clean = ' '.join(w for i, w in enumerate(ref_words) if i not in ref_remove_idx)
    h_clean = ' '.join(w for i, w in enumerate(hyp_words) if i not in hyp_remove_idx)

    return gt_clean, h_clean

# Fixing the ground truth (OUTDATED)
# Clean data for category Dokter Patient directory
def dok_pat(t):
    t = t.split('\n')
    t = [line.strip for line in t]
    t = " ".join([t[i*2+1] for i in range(int(len(t)/2))])
    return t


# Clean data for category Pedagogische gesprekken
def ped_ges(t):
    t = t.split('\n')
    t = [line.strip() for line in t]
    st = ''
    for line in t:
        st += line[line.find(':')+1:]
    return st

#
clean_func = {'Dokter Patient': ped_ges, 'Psychologische gespreksvoering': ped_ges, 'Test': dok_pat, 'interviews': ped_ges}

if __name__ == "__main__":
    pass
