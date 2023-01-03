import logging
import os
from pathlib import Path

import NLPyPort as pyport
import pandas as pd
from NLPyPort.FullPipeline import TokPort_config_file

logger = logging.getLogger('HumorRecognitionPT')


def preprocess_data(corpus: pd.DataFrame) -> pd.DataFrame:
    logger.info('Starting preprocessing')
    pyport.load_config()

    def tokenizer(sent):
        return pyport.tokenize_from_string(sent, TokPort_config_file)

    def tagger(sent):
        return pyport.tag(sent)[0]

    def lemmatizer(sent):
        return pyport.lematizador_normal(sent['Tokens'], sent['POS Tags'])

    def ner(sent, model):
        X = list()
        for i in range(len(sent['Tokens'])):
            feats = pyport.features(sent['Tokens'], sent['POS Tags'],
                                    sent['Lemma'], i, [True, True])
            X.append(feats)
        prediction = model.predict([X])
        return prediction[0]

    logger.info('Tokenizing corpus')
    corpus['Tokens'] = corpus['Text'].apply(tokenizer)
    logger.info('Tagging corpus')
    corpus['POS Tags'] = corpus['Tokens'].apply(tagger)
    logger.info('Lemmatizing corpus')
    corpus['Lemma'] = corpus.apply(lemmatizer, axis=1)
    logger.info('Running Named Entity Recognition')
    pyport_path = Path(os.path.dirname(pyport.__file__))
    model_path = pyport_path / 'CRF/trainedModels/harem.pickle'
    model = pyport.load_model(model_path)
    corpus['NER'] = corpus.apply(ner, axis=1, args=(model,))

    logger.debug(f'Preprocessed corpus\n\n{corpus}')
    return corpus
