import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.data import read_file, preprocess_data


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i',
                        help='Corpus in TSV format.',
                        required=True, type=Path)
    parser.add_argument('--verbose', '-v',
                        action='count', default=0,
                        help='Print informatino and debugging messages')
    return parser.parse_args()


def config_logger(verbose_level):
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


if __name__ == '__main__':
    args = parse_args()
    config_logger(args.verbose)
    main(args)
