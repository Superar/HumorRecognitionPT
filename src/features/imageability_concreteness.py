import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger('HumorRecognitionPT')


def calculate_values(lemmas: list[str], mwp: pd.DataFrame):
    lemmas = pd.Series(lemmas)
    lemmas = lemmas[lemmas.isin(mwp['Word (Portuguese)'])]

    values = mwp.loc[mwp['Word (Portuguese)'].isin(lemmas),
                     ['Imag_M', 'Conc_M']]
    return values.mean()


def calculate_imageability_concreteness(corpus: pd.DataFrame,
                                        mwp: Path) -> pd.DataFrame:
    logger.info('Reading Minho World Pool lexicon')
    mwp = pd.read_json(mwp)
    logger.debug(f'Minho World Pool\n\n{mwp}')

    logger.info('Calculating imageability and concreteness features')
    values = corpus['Lemma'].apply(calculate_values, mwp=mwp).fillna(0)
    values.columns = ['Average imageability', 'Average concreteness']
    logger.debug(f'Imageability and concreteness\n\n{values}')

    return values
