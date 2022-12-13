from pathlib import Path
from NLPyPort.FullPipeline import TokPort_config_file

import NLPyPort as pyport
import pandas as pd

import logging

logger = logging.getLogger('HumorRecognitionPT')


def read_file(filename: Path) -> pd.DataFrame:
    logger.info(f'Reading file {filename}')
    df = pd.read_csv(filename, sep='\t',
                     names=['Text', 'Label'])
    logger.debug(f'\n\n{df}')
    return df


def preprocess_data(corpus: pd.DataFrame) -> pd.DataFrame:
    logger.info('Starting preprocessing')
    pyport.load_config()

    def tokenizer(sent):
        return pyport.tokenize_from_string(sent, TokPort_config_file)

    def tagger(sent):
        return pyport.tag(sent)[0]

    def lemmatizer(sent):
        return pyport.lematizador_normal(sent['Tokens'], sent['POS Tags'])

    logger.info('Tokenizing corpus')
    corpus['Tokens'] = corpus['Text'].apply(tokenizer)
    logger.info('Tagging corpus')
    corpus['POS Tags'] = corpus['Tokens'].apply(tagger)
    logger.info('Lemmatizing corpus')
    corpus['Lemma'] = corpus.apply(lemmatizer, axis=1)

    logger.info('Preprocessing done')
    return corpus
