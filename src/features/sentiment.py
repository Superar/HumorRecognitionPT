import logging
from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.string import to_lower

logger = logging.getLogger('HumorRecognitionPT')


def more_positive_or_negative(instance: pd.Series) -> int:
    negative = instance['Negative Sentiment']
    positive = instance['Positive Sentiment']
    if negative > positive:
        return 0
    elif negative == positive:
        return 1
    else:
        return 2


def calculate_sentiment(corpus: pd.DataFrame, lexicon: Path) -> tuple[int, int]:
    corpus = corpus.copy()  # Prevent inplace operations
    lexicon = pd.read_json(lexicon, orient='index')
    tokens = corpus['Tokens'].map(to_lower)
    logger.debug(f'Sentiment lexicon\n\n{lexicon}')

    logger.info('Retrieving tokens in the lexicon')
    toks_in_lex = tokens.map(
        lambda x: x[x.isin(lexicon.index)])
    logger.info('Retrieving token sentiments')
    corpus['Sentiments'] = toks_in_lex.map(
        lambda x: lexicon.loc[x, 'Polarity'])
    logger.info('Computing sentiment features')
    corpus['Positive Sentiment'] = corpus['Sentiments'].map(
        lambda x: np.count_nonzero(x == 1))
    corpus['Negative Sentiment'] = corpus['Sentiments'].map(
        lambda x: np.count_nonzero(x == -1))
    corpus['Negative > Positive'] = corpus.apply(
        more_positive_or_negative, axis=1)

    sentiment_features = corpus[['Positive Sentiment',
                                 'Negative Sentiment',
                                 'Negative > Positive']]
    logger.debug(f'Sentiment features\n\n{sentiment_features}')
    return sentiment_features
