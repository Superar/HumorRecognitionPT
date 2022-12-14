import logging

import pandas as pd
from scipy.sparse._csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger('HumorRecognitionPT')


def calculate_tfidf(corpus: pd.DataFrame, ngram: str = '1+2+3') -> tuple[TfidfVectorizer, csr_matrix]:
    global dummy_tokenizer
    def dummy_tokenizer(toks): return toks

    split_ngram = [int(n) for n in ngram.split('+')]
    ngram_range = (split_ngram[0], split_ngram[-1])
    logger.debug(f'Using ngram range: {ngram_range}')

    logger.info('Calculating TF-IDF counts')
    vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer,
                                 preprocessor=dummy_tokenizer,
                                 ngram_range=ngram_range)
    counts = vectorizer.fit_transform(corpus['Tokens'])
    logger.info('TF-IDF done')
    logger.debug(f'TF-IDF matrix shape: {counts.shape}')
    return vectorizer, counts
