import logging
from itertools import combinations
from pathlib import Path

import pandas as pd

logger = logging.getLogger('HumorRecognitionPT')


def number_of_antonymy_pairs(lemmas: list[str], lexicon: pd.DataFrame) -> int:
    pairs = pd.DataFrame(combinations(lemmas, 2),
                         columns=['Lemma 1', 'Lemma 2'])
    forward_rel = pairs.merge(lexicon, left_on=['Lemma 1', 'Lemma 2'],
                              right_on=['Argument 1', 'Argument 2'])
    backward_rel = pairs.merge(lexicon, left_on=['Lemma 1', 'Lemma 2'],
                               right_on=['Argument 2', 'Argument 1'])

    # Work only with the pairs themselves (ignore argument order)
    on = ['Lemma 1', 'Lemma 2']
    forward_rel = forward_rel[on]
    backward_rel = backward_rel[on]
    # Remove backward pairs that are already in the forward dataframe
    backward_rel = (forward_rel.merge(backward_rel, how='right', indicator=True)
                    .query('_merge == "right_only"')
                    .drop(columns=['_merge']))
    all_pairs = pd.concat([forward_rel, backward_rel])
    return len(all_pairs)


def calculate_antonym(corpus: pd.DataFrame, lexicon: Path):
    logger.info(f'Reading antonym lexicon from {lexicon}')
    lexicon = pd.read_json(lexicon)
    logger.debug(f'Antonym lexicon\n\n{lexicon}')
    logger.info('Counting antonym pairs')
    corpus['Antonym'] = corpus['Lemma'].map(
        lambda x: number_of_antonymy_pairs(x, lexicon))
    logger.debug(f'Antonym features\n\n{corpus["Antonym"].describe()}')

    return corpus['Antonym']
