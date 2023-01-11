import logging
from argparse import ArgumentParser, Namespace
from pathlib import Path

from src import commands
from src.methods import clemencio


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
                                   required=False, type=Path)
    parser_preprocess.set_defaults(command=commands.preprocess)

    # feature_extraction
    parser_feature = subparsers.add_parser('feature-extraction',
                                           aliases=['feat'])
    parser_feature.add_argument('--input', '-i',
                                help='Preprocessed corpus in JSON format.',
                                required=True, type=Path)
    parser_feature.add_argument('--output', '-o',
                                help='Directory path to save count models and feature matrix.',
                                required=False, type=Path)
    parser_feature.add_argument('--tfidf',
                                help='Flag to use TF-IDF counts',
                                required=False, action='store_true',
                                default=False)
    parser_feature.add_argument('--vectorizer',
                                help='Path to the pickled TfIdfVectorizer to use for TF-IDF counts',
                                required=False, type=Path,
                                default=None)
    parser_feature.add_argument('--ngram', '-n',
                                help='Which n-gram configuration to use to calculate the TF-IDF counts',
                                required=False, type=str,
                                choices=['1', '2', '3',
                                         '1+2', '2+3',
                                         '1+2+3'],
                                default='1+2+3')
    parser_feature.add_argument('--sentlex',
                                help='Path to the sentiment lexicon in JSON format.',
                                required=False, type=Path,
                                default=None)
    parser_feature.add_argument('--slang',
                                help='Path to the slang lexicon in JSON format.',
                                required=False, type=Path,
                                default=None)
    parser_feature.add_argument('--alliteration',
                                help='Flag to use alliteration features',
                                required=False, action='store_true',
                                default=False)
    parser_feature.add_argument('--antonym',
                                help='Path to the antonym triples lexicon in JSON format.',
                                required=False, type=Path,
                                default=None)
    parser_feature.add_argument('--embeddings',
                                help='Path to the word embeddings file in Gensim format.',
                                required=False, type=Path,
                                default=None)
    parser_feature.add_argument('--mwp',
                                help=('Path to the Minho World Pool lexicon for imageability '
                                      'and concreteness features in JSON format.'),
                                required=False, type=Path,
                                default=None)
    parser_feature.add_argument('--ner',
                                help='Flag to use NER features',
                                required=False, action='store_true',
                                default=False)
    parser_feature.add_argument('--ambiguity',
                                help='Flag to use ambiguity features from OpenWordNet-PT',
                                required=False, action='store_true',
                                default=False)
    parser_feature.set_defaults(command=commands.feature_extraction)

    # Method parsers
    clemencio.add_parser(subparsers)

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
    args.command(args)


if __name__ == '__main__':
    args = parse_args()
    config_logger(args.verbose)
    main(args)
