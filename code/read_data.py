import numpy as np
import re


def get_sentences(path):
    """
    Return a list of sentences from the given path, ignoring line numbers.
    """
    with open(path, 'r') as f:
        lines = f.readlines()

    # replace tabs (optionally preceded with a space) with a space
    lines = [re.sub(r' ?\t', ' ', line) for line in lines]

    regex = re.compile(r"""(\d+) # the prefixed line number
                           \     # followed by a space
                           (.*)  # the rest of the sentence""", re.VERBOSE)

    # second regex group is the part we're interested in
    return [re.match(regex, line).groups()[1] for line in lines]


def sentences_to_word_encodings(sentences):
    """
    Convert a list of loaded sentences to a dictionary mapping words to their
    representational vectors.
    """
    word_lists = [line.split(' ') for line in sentences]
    words = set(word for word_list in word_lists for word in word_list)

    mapping = {}
    for i, word in enumerate(words):
        encoding = [0] * len(words)
        encoding[i] = 1

        mapping[word] = encoding

    return mapping
