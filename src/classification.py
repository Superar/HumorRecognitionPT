import logging

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC

logger = logging.getLogger('HumorRecognitionPT')


def train_model(X: pd.DataFrame, y: pd.Series, method: str = 'SVC'):
    logger.info('Start training')

    if method == 'SVC':
        model = SVC(probability=True)
    elif method == 'SVCLinear':
        model = SVC(kernel='linear', probability=True)
    elif method == 'MultinomialNB':
        model = MultinomialNB()
    elif method == 'GaussianNB':
        model = GaussianNB()
    elif method == 'RandomForest':
        model = RandomForestClassifier()

    logger.debug(f'Training model: {model}')
    X.columns = X.columns.astype(str)
    model.fit(X, y)
    logger.info('Training done')
    return model
