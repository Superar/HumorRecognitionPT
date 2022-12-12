from pathlib import Path
from NLPyPort.FullPipeline import TokPort_config_file

import NLPyPort as pyport
import pandas as pd


def read_file(filename: Path) -> pd.DataFrame:
    return pd.read_csv(filename, sep='\t',
                       names=['Sentence', 'Label'])


def preprocess_data(corpus: pd.DataFrame) -> pd.DataFrame:
    pyport.load_config()

    def tokenizer(sent):
        return pyport.tokenize_from_string(sent, TokPort_config_file)

    def tagger(sent):
        return pyport.tag(sent)[0]

    def lemmatizer(sent):
        return pyport.lematizador_normal(sent['Tokens'], sent['POS Tags'])

    corpus['Tokens'] = corpus['Sentence'].apply(tokenizer)
    corpus['POS Tags'] = corpus['Tokens'].apply(tagger)
    corpus['Lemma'] = corpus.apply(lemmatizer, axis=1)

    return corpus
