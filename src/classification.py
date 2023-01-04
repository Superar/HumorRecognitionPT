import logging

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.ensemble import RandomForestClassifier

logger = logging.getLogger('HumorRecognitionPT')


def train_model(X, y, method='SVC'):
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
    model.fit(X, y)
    logger.info('Training done')
    return model
