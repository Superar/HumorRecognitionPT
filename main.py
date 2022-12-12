from argparse import ArgumentParser, Namespace
from pathlib import Path

from src.data import read_file, preprocess_data


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument('--input', '-i',
                        help='Corpus in TSV format.',
                        required=True, type=Path)
    return parser.parse_args()


def main(args):
    corpus = read_file(args.input) 
    corpus = preprocess_data(corpus)


if __name__ == '__main__':
    args = parse_args()
    main(args)
