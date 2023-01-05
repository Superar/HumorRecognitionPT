import logging
from itertools import combinations
from pathlib import Path

import pandas as pd
from gensim.models import KeyedVectors

logger = logging.getLogger('HumorRecognitionPT')


def calculate_similarity(tokens: list[str],
                         embeddings: KeyedVectors) -> pd.Series:
    tokens = [t for t in tokens if t in embeddings]
    pairs = pd.DataFrame(combinations(tokens, 2),
                         columns=['Token 1', 'Token 2'])
    similarity = pairs.apply(lambda x: embeddings.similarity(x['Token 1'], x['Token 2']),
                             axis='columns')
    similarity += 1  # Avoid negative values
    return similarity.mean(), similarity.min()


def calculate_embeddings(corpus: pd.DataFrame,
                         embeddings: Path) -> pd.DataFrame:
    logger.info(f'Loading embeddings from {embeddings}')
    embeddings = KeyedVectors.load_word2vec_format(embeddings)
    logger.info(f'{len(embeddings)} embeddings loaded')

    tokens = corpus['Tokens']
    logger.info(f'Computing out-of-vocabulary words')
    corpus['OoV'] = tokens.map(
        lambda x: len([t for t in x if t not in embeddings]))
    logger.debug(f'Out-of-vocabulary\n\n{corpus["OoV"]}')

    logger.info('Calculating similarity')
    similarity = tokens.map(lambda x: calculate_similarity(x, embeddings))
    similarity = pd.DataFrame(similarity.to_list(),
                              columns=['Average similarity', 'Minimum similarity'],
                              index=corpus.index)
    logger.debug(f'Similarity\n\n{similarity}')

    return similarity
