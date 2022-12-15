import logging
from pathlib import Path

import numpy as np
import pandas as pd
from src.utils.string import to_lower

logger = logging.getLogger('HumorRecognitionPT')


def calculate_sentiment(corpus: pd.DataFrame, lexicon: Path) -> tuple[int, int]:
    lexicon = pd.read_json(lexicon, orient='index')
    tokens = corpus['Tokens'].map(to_lower)
    logger.debug(f'Sentiment lexicon\n\n{lexicon}')

    toks_in_lex = tokens.map(lambda x: lexicon.index.intersection(x))
    sentiments = toks_in_lex.map(lambda x: lexicon.loc[x, 'Polarity'])
    positive = sentiments.map(lambda x: np.count_nonzero(x == 1))
    negative = sentiments.map(lambda x: np.count_nonzero(x == -1))
    return positive, negative
