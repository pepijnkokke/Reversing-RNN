import numpy as np
import itertools
import re


def get_sentences(path):
    """
    Return a list of sentences from the given path, ignoring line numbers.
    """
    with open(path, 'r') as f:
        lines = f.readlines()
    return [re.sub(r"\s+",' ',l) for l in lines]


def sentence_to_tokens(sentence):
    return re.findall(r"[\w']+|[.,!?;]", sentence)


def sentences_to_word_encodings(sentences):
    """
    Convert a list of loaded sentences to a dictionary mapping words to their
    representational vectors.
    """
    sentences = [sentence_to_tokens(s) for s in sentences]
    tokens    = list(set(itertools.chain.from_iterable(sentences)))
    K         = len(tokens)

    encoding = {}
    for i, token in enumerate(tokens):
        token_encoding    = [0] * K
        token_encoding[i] = 1
        encoding[token]   = token_encoding
    return (K, encoding)
