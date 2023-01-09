from argparse import Namespace
from logging import getLogger
from src.data import preprocess_data
import pandas as pd


logger = getLogger('HumorRecognitionPT')


def preprocess(args: Namespace):
    logger.info(f'Loading file {args.input}')
    corpus = pd.read_json(args.input)
    logger.debug(f'\n\n{corpus}')

    corpus = preprocess_data(corpus)
    if args.output:
        corpus.to_json(args.output, orient='records',
                       force_ascii=False, indent=4)
