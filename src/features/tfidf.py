import logging
from pathlib import Path
from typing import Union

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

logger = logging.getLogger('HumorRecognitionPT')


def calculate_tfidf(corpus: pd.DataFrame,
                    ngram: str = '1+2+3',
                    vectorizer_file: Union[Path, None] = None,
                    max_features: int = 1000) -> tuple[TfidfVectorizer, pd.DataFrame]:
    global dummy_tokenizer
    def dummy_tokenizer(toks): return toks

    split_ngram = [int(n) for n in ngram.split('+')]
    ngram_range = (split_ngram[0], split_ngram[-1])
    logger.debug(f'Using ngram range: {ngram_range}')

    if vectorizer_file is not None:
        logger.debug(f'Reading vectorizer from file: {vectorizer_file}')
        with vectorizer_file.open('rb') as file_:
            vectorizer = pickle.load(file_)
    else:
        logger.debug('Creating new vectorizer')
        vectorizer = TfidfVectorizer(tokenizer=dummy_tokenizer,
                                     preprocessor=dummy_tokenizer,
                                     ngram_range=ngram_range,
                                     min_df=2, max_df=0.75,
                                     max_features=max_features)
        logger.debug('Fitting new vectorizer')
        vectorizer = vectorizer.fit(corpus['Tokens'])
        logger.debug('Vectorizer fit complete')

    logger.info('Calculating TF-IDF counts')
    counts = vectorizer.transform(corpus['Tokens'])
    logger.info('TF-IDF done')
    logger.debug(f'TF-IDF matrix shape: {counts.shape}')
    tf_idf = pd.DataFrame(counts.toarray(),
                          columns=vectorizer.get_feature_names_out(),
                          index=corpus.index)
    logger.debug(f'TF-IDF summary\n\n{tf_idf.describe()}')
    return vectorizer, tf_idf
