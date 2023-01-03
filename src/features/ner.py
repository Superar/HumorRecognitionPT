import logging

import pandas as pd

logger = logging.getLogger('HumorRecognitionPT')


def number_of_tags_per_type(sent: list[str], types: pd.Series) -> pd.Series:
    sent = pd.Series(sent)

    clean_sent = sent.str.replace(r'B-(\w+)', r'\1')
    ner_counts = types.map(clean_sent.value_counts()).fillna(0)
    return ner_counts


def calculate_ner(corpus: pd.DataFrame) -> pd.DataFrame:
    ner_tags = corpus['NER']
    types = pd.Series(['OBRA', 'ACONTECIMENTO', 'ORGANIZACAO', 'OUTRO',
                       'ABSTRACCAO', 'TEMPO', 'VALOR', 'PESSOA', 'COISA', 'LOCAL'])

    logger.info('Counting number of NER tags')
    ner_counts = ner_tags.apply(number_of_tags_per_type, args=(types,))
    ner_counts.columns = 'NER count ' + types
    ner_counts['NER count sum'] = ner_counts.sum(axis='columns')
    logger.debug(f'NER counts\n\n{ner_counts.describe()}')

    return ner_counts
