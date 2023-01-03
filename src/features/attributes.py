import logging
from pathlib import Path

import pandas as pd
from src.features import (calculate_alliteration, calculate_antonym,
                          calculate_sentiment, calculate_slang,
                          calculate_tfidf, calculate_embeddings,
                          calculate_imageability_concreteness,
                          calculate_ner)

logger = logging.getLogger('HumorRecognitionPT')


def calculate_features(corpus: pd.DataFrame,
                       ngram: str = '1+2+3',
                       sentiment_lexicon: Path = None,
                       slang_lexicon: Path = None,
                       alliteration: bool = False,
                       antonym_lexicon: Path = None,
                       embeddings: Path = None,
                       mwp: Path = None,
                       ner: bool = False):
    vectorizer, features = calculate_tfidf(corpus, ngram)
    if sentiment_lexicon:
        sentiment_features = calculate_sentiment(corpus, sentiment_lexicon)
        features = features.join(sentiment_features)
    if slang_lexicon:
        slang_features = calculate_slang(corpus, slang_lexicon)
        features = features.join(slang_features)
    if alliteration:
        alliteration_features = calculate_alliteration(corpus)
        features = features.join(alliteration_features)
    if antonym_lexicon:
        antonym_features = calculate_antonym(corpus, antonym_lexicon)
        features = features.join(antonym_features)
    if embeddings:
        embeddings_features = calculate_embeddings(corpus, embeddings)
        features = features.join(embeddings_features)
    if mwp:
        mwp_features = calculate_imageability_concreteness(corpus, mwp)
        features = features.join(mwp_features)
    if ner:
        ner_features = calculate_ner(corpus)
        features = features.join(ner_features)
    features['Label'] = corpus['Label']

    return vectorizer, features
