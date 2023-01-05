import logging

import pandas as pd
from nltk.corpus import wordnet as wn

logger = logging.getLogger('HumorRecognitionPT')


def number_of_senses(lemmas: list[str]) -> pd.Series:
    lemmas = pd.Series(lemmas)
    synsets = lemmas.apply(wn.synsets, lang='por')
    num_senses = synsets.str.len().replace(0, 1)
    return num_senses.mean(), num_senses.max()


def calculate_ambiguity(corpus: pd.DataFrame) -> pd.DataFrame:
    lemmas = corpus['Lemma']
    logger.info('Calculating number of senses')
    num_senses = lemmas.apply(number_of_senses)
    num_senses = pd.DataFrame(num_senses.to_list(),
                              columns=['Average number of senses',
                                       'Maximum number of senses'],
                              index=corpus.index)
    logger.debug(f'Number of senses\n\n{num_senses}')
    return num_senses
