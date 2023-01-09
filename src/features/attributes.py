import logging
from pathlib import Path
from typing import Union

import pandas as pd
from src.features import (calculate_alliteration, calculate_ambiguity,
                          calculate_antonym, calculate_embeddings,
                          calculate_imageability_concreteness, calculate_ner,
                          calculate_sentiment, calculate_slang,
                          calculate_tfidf)
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger('HumorRecognitionPT')


def calculate_features(corpus: pd.DataFrame,
                       tfidf: bool = False,
                       vectorizer_path: Union[Path, None] = None,
                       ngram: str = '1+2+3',
                       sentiment_lexicon: Union[Path, None] = None,
                       slang_lexicon: Union[Path, None] = None,
                       alliteration: bool = False,
                       antonym_lexicon: Union[Path, None] = None,
                       embeddings: Union[Path, None] = None,
                       mwp: Union[Path, None] = None,
                       ner: bool = False,
                       ambiguity: bool = False) -> tuple[Union[TfidfVectorizer, None], pd.DataFrame]:
    features = pd.DataFrame(corpus['Label'])

    vectorizer = None
    if tfidf:
        vectorizer, tfidf_features = calculate_tfidf(corpus, ngram,
                                                     vectorizer_path)
        features = features.join(tfidf_features)
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
    if ambiguity:
        ambiguity_features = calculate_ambiguity(corpus)
        features = features.join(ambiguity_features)

    return vectorizer, features
