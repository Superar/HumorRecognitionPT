from argparse import Namespace
from logging import getLogger
import pandas as pd
from src.classification import train_model
import joblib


logger = getLogger('HumorRecognitionPT')


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
