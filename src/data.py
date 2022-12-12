from pathlib import Path

import NLPyPort as pyport
import pandas as pd


def read_file(filename: Path) -> pd.DataFrame:
    return pd.read_csv(filename, sep='\t',
                       names=['Sentence', 'Label'])


def preprocess_data(corpus: pd.DataFrame) -> pd.DataFrame:
    pyport.FullPipeline.load_config()

    def tokenizer(sent):
        tok_config = pyport.FullPipeline.TokPort_config_file
        return pyport.tokenize_from_string(sent, tok_config)

    def tagger(sent):
        return pyport.FullPipeline.tag(sent)[0]

    def lemmatizer(sent): return pyport.lematizador_normal(
        sent['Tokens'], sent['POS Tags'])

    corpus['Tokens'] = corpus['Sentence'].apply(tokenizer)
    corpus['POS Tags'] = corpus['Tokens'].apply(tagger)
    corpus['Lemma'] = corpus.apply(lemmatizer, axis=1)

    return corpus
