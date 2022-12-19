import logging
from pathlib import Path

import pandas as pd
from src.features import (calculate_alliteration, calculate_sentiment,
                          calculate_slang, calculate_tfidf)

logger = logging.getLogger('HumorRecognitionPT')


def calculate_features(corpus: pd.DataFrame,
                       ngram: str = '1+2+3',
                       sentiment_lexicon: Path = None,
                       slang_lexicon: Path = None,
                       alliteration: bool = False):
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
    features['Label'] = corpus['Label']

    return vectorizer, features
