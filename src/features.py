import logging

import pandas as pd
from scipy.sparse._csr import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger('HumorRecognitionPT')


def calculate_tfidf(corpus: pd.DataFrame) -> tuple[TfidfVectorizer, csr_matrix]:
    logger.info('Calculating TF-IDF counts')
    vectorizer = TfidfVectorizer(analyzer=lambda toks: toks)
    counts = vectorizer.fit_transform(corpus['Tokens'])
    logger.info('TF-IDF done')
    logger.debug(f'TF-IDF matrix shape: {counts.shape}')
    return vectorizer, counts
