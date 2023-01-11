from argparse import Namespace
from logging import getLogger
from pathlib import Path

import joblib
import pandas as pd
from src.classification import train_model

logger = getLogger('HumorRecognitionPT')


def add_parser(subparsers):
    clemencio_parser = subparsers.add_parser('clemencio')
    clemencio_subparsers = clemencio_parser.add_subparsers()

    # train
    parser_train = clemencio_subparsers.add_parser('train')
    parser_train.add_argument('--input', '-i',
                              help='Training data in HDF5 format.',
                              required=True, type=Path)
    parser_train.add_argument('--output', '-o',
                              help='Directory path to save the model.',
                              required=False, type=Path)
    parser_train.add_argument('--method', '-m',
                              help='Which classification approach to use from Scikit-learn',
                              required=False, type=str,
                              choices=['SVC', 'SVCLinear', 'MultinomialNB',
                                       'GaussianNB', 'RandomForest'],
                              default='SVC')
    parser_train.set_defaults(command=train)

    # test
    parser_test = clemencio_subparsers.add_parser('test')
    parser_test.add_argument('--input', '-i',
                             help='Test data in JSON format.',
                             required=True, type=Path)
    parser_test.add_argument('--model', '-m',
                             help='Model directory path.',
                             required=True, type=Path)
    parser_test.add_argument('--output', '-o',
                             help='Path to the file to save the predictions in JSON format.',
                             required=False, type=Path,
                             default=None)
    parser_test.set_defaults(command=test)


def train(args: Namespace):
    logger.info(f'Loading file {args.input}')
    df = pd.read_hdf(args.input)
    logger.debug(f'\n\n{df}')

    X = df.drop(columns=['Label'])
    y = df['Label']
    model = train_model(X, y, args.method)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        model_path = args.output / 'model.joblib'
        logger.info(f'Saving model to {model_path}')
        joblib.dump(model, model_path)


def test(args: Namespace):
    logger.info(f'Loading file {args.input}')
    df = pd.read_hdf(args.input)
    logger.debug(f'\n\n{df}')

    model_path = args.model / 'model.joblib'
    logger.info(f'Loading model {model_path}')
    model = joblib.load(model_path)
    logger.debug(f'Model loaded: {model}')

    X = df.drop(columns=['Label'])
    results = df[['Label']]
    results['Prediction'] = model.predict(X)
    logger.debug(f'\n\n{results}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving predictions to {args.output}')
        results.to_json(args.output, force_ascii=False, indent=4)
        logger.info('Predictions saved')
