import logging

from sklearn.svm import SVC

logger = logging.getLogger('HumorRecognitionPT')


def train_model(X, y, type='SVC'):
    logger.info('Start training')
    model = SVC()
    model.fit(X, y)
    logger.info('Training done')
    return model
