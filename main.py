import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from src.classification import train_model
from src.data import preprocess_data, read_file
from src.features import calculate_tfidf


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i',
                        help='Corpus in TSV format.',
                        required=True, type=Path)
    parser.add_argument('--verbose', '-v',
                        action='count', default=0,
                        help='Print informatino and debugging messages')
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
    corpus = read_file(args.input)
    corpus = preprocess_data(corpus)
    _, features = calculate_tfidf(corpus)
    logger.info('Split train and test')
    train_X, test_X, train_y, test_y = train_test_split(features,
                                                        corpus['Label'],
                                                        test_size=0.3)
    logger.debug(f'train_X: {train_X.shape}; train_y: {train_y.shape}')
    logger.debug(f'test_X: {test_X.shape}; test_y: {test_y.shape}')

    model = train_model(train_X, train_y)
    results = model.predict(test_X)
    evaluation = classification_report(test_y, results)
    logger.info(f'\n\n{evaluation}')


if __name__ == '__main__':
    args = parse_args()
    config_logger(args.verbose)
    main(args)
