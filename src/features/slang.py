import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger('HumorRecognitionPT')


def calculate_slang(corpus: pd.DataFrame, lexicon: Path):
    corpus = corpus.copy()  # Prevent inplace operations
    lexicon = pd.read_json(lexicon, orient='index', typ='series')
    toks_in_lex = corpus["Lemma"].map(lambda x: pd.Series(x).isin(lexicon))
    corpus['Slangs'] = toks_in_lex.map(lambda x: np.count_nonzero(x))

    logger.debug(f'Slang lexicon\n\n{lexicon}')
    logger.debug(f'Slangs\n\n{corpus["Slangs"]}')

    return corpus['Slangs']
