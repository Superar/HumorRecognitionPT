import logging
import pickle
import time
from argparse import ArgumentParser, Namespace
from pathlib import Path

import joblib
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.classification import train_model
from src.data import preprocess_data
from src.features import calculate_tfidf


def parse_args() -> Namespace:
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    parser.add_argument('--verbose', '-v',
                        action='count', default=0,
                        help='Print information and debugging messages')

    # preprocess
    parser_preprocess = subparsers.add_parser('preprocess')
    parser_preprocess.add_argument('--input', '-i',
                                   help='Corpus file in JSON format.',
                                   required=True, type=Path)
    parser_preprocess.add_argument('--output', '-o',
                                   help='File path to save preprocessed data in JSON format.',
                                   required=True, type=Path)

    # feature_extraction
    parser_feature = subparsers.add_parser('feature-extraction',
                                           aliases=['feat'])
    parser_feature.add_argument('--input', '-i',
                                help='Preprocessed corpus in JSON format.',
                                required=True, type=Path)
    parser_feature.add_argument('--output', '-o',
                                help='Directory path to save count models and feature matrix.',
                                required=True, type=Path)
    parser_feature.add_argument('--ngram', '-n',
                                help='Which n-gram configuration to use to calculate the TF-IDF counts',
                                required=False, type=str,
                                choices=['1', '2', '3',
                                         '1+2', '2+3',
                                         '1+2+3'],
                                default='1+2+3')

    # train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--input', '-i',
                              help='Training data in JSON format.',
                              required=True, type=Path)
    parser_train.add_argument('--output', '-o',
                              help='Directory path to save the model.',
                              required=True, type=Path)

    # test
    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--input', '-i',
                             help='Test data in JSON format.',
                             required=True, type=Path)
    parser_test.add_argument('--model', '-m',
                             help='Model directory path.',
                             required=True, type=Path)
    return parser.parse_args()


def config_logger(verbose_level: int):
    global logger
    logger = logging.getLogger('HumorRecognitionPT')
    ch = logging.StreamHandler()

    if verbose_level == 1:
        logger.setLevel(logging.INFO)
    elif verbose_level >= 1:
        logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s: %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def main(args):
    if args.command == 'preprocess':
        logger.info(f'Loading file {args.input}')
        corpus = pd.read_json(args.input)
        logger.debug(f'\n\n{corpus}')

        corpus = preprocess_data(corpus)
        corpus.to_json(args.output, orient='records',
                       force_ascii=False, indent=4)
    elif args.command == 'feature-extraction' or args.command == 'feat':
        corpus = pd.read_json(args.input)
        vectorizer, features = calculate_tfidf(corpus, args.ngram)
        df = pd.DataFrame(features.toarray())
        df['Label'] = corpus['Label']

        args.output.mkdir(parents=True, exist_ok=True)
        vectorizer_path = args.output / 'vectorizer.pkl'
        data_path = args.output / 'data.json'
        logger.info(f'Saving vectorizer to {vectorizer_path}')
        with (vectorizer_path).open('wb') as file_:
            pickle.dump(vectorizer, file_)
        logger.info(f'Saving data to {data_path}')
        df.to_json(data_path,
                   orient='records',
                   force_ascii=False,
                   indent=4)
    elif args.command == 'train':
        logger.info(f'Loading file {args.input}')
        df = pd.read_json(args.input)
        logger.debug(f'\n\n{df}')

        X = df.drop(columns=['Label'])
        y = df['Label']
        model = train_model(X, y)

        args.output.mkdir(parents=True, exist_ok=True)
        model_path = args.output / 'model.joblib'
        logger.info(f'Saving model to {model_path}')
        joblib.dump(model, model_path)
    elif args.command == 'test':
        logger.info(f'Loading file {args.input}')
        df = pd.read_json(args.input)
        logger.debug(f'\n\n{df}')

        model_path = args.model / 'model.joblib'
        logger.info(f'Loading model {model_path}')
        model = joblib.load(model_path)
        logger.debug(f'Model loaded: {model}')

        X = df.drop(columns=['Label'])
        y = df['Label']
        results = model.predict(X)
        evaluation = classification_report(y, results)
        logger.info(f'\n\n{evaluation}')


if __name__ == '__main__':
    args = parse_args()
    config_logger(args.verbose)
    main(args)
