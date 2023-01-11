import pickle
from argparse import Namespace
from logging import getLogger

import pandas as pd
from src.features import calculate_features

logger = getLogger('HumorRecognitionPT')


def feature_extraction(args: Namespace):
    corpus = pd.read_json(args.input)
    logger.debug(f'Corpus\n\n{corpus}')
    vectorizer, features = calculate_features(corpus,
                                              args.tfidf,
                                              args.vectorizer,
                                              args.ngram,
                                              args.sentlex,
                                              args.slang,
                                              args.alliteration,
                                              args.antonym,
                                              args.embeddings,
                                              args.mwp,
                                              args.ner,
                                              args.ambiguity)
    logger.debug(f'Feature matrix\n\n{features}')

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        # Save vectorizer
        if args.tfidf and args.vectorizer is None:
            vectorizer_path = args.output / 'vectorizer.pkl'
            logger.info(f'Saving vectorizer to {vectorizer_path}')
            with (vectorizer_path).open('wb') as file_:
                pickle.dump(vectorizer, file_)

        # Save features
        data_path = args.output / 'data.hdf5'
        logger.info(f'Saving data to {data_path}')
        features.to_hdf(data_path, key='df', mode='w')
