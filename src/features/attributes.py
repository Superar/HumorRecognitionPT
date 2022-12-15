import logging
from pathlib import Path

import pandas as pd
from src.features import calculate_sentiment, calculate_tfidf

logger = logging.getLogger('HumorRecognitionPT')


def calculate_features(corpus: pd.DataFrame,
                       ngram: str = '1+2+3',
                       sentiment_lexicon: Path = None):
    vectorizer, features = calculate_tfidf(corpus, ngram)
    if sentiment_lexicon:
        positive_counts, negative_counts = calculate_sentiment(
            corpus, sentiment_lexicon)
        features['Positive Sentiment'] = positive_counts
        features['Negative Sentiment'] = negative_counts
    features['Label'] = corpus['Label']

    return vectorizer, features
