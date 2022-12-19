import logging

import pandas as pd
from nltk.util import ngrams
from src.utils.string import to_lower

logger = logging.getLogger('HumorRecognitionPT')


def highest_character_ngram(tokens: pd.Series, n: int):
    char_ngrams = tokens.apply(
        lambda x: list(ngrams(x, n)))  # Series of list of ngrams
    char_ngrams = char_ngrams.explode(0)  # Flatten into a Series of ngrams
    highest_ngram_count = char_ngrams.value_counts().max()
    return highest_ngram_count


def calculate_alliteration(corpus: pd.DataFrame):
    corpus = corpus.copy()  # Prevent inplace operations
    tokens = corpus['Tokens'].map(to_lower)

    new_columns = list()
    for n in range(1, 5):
        logger.info(f'Calculating alliteration feature for {n}-grams')
        column = f'Alliteration - {n}-gram'
        highest_ngram_count = tokens.apply(highest_character_ngram, args=(n,))
        corpus[column] = highest_ngram_count
        new_columns.append(column)
    
    logger.debug(f'Alliteration features\n\n{corpus[new_columns]}')
    return corpus[new_columns]
