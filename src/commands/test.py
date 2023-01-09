from argparse import Namespace
from logging import getLogger
import pandas as pd
import joblib


logger = getLogger('HumorRecognitionPT')


def test(args: Namespace):
    logger.info(f'Loading file {args.input}')
    df = pd.read_hdf(args.input)
    logger.debug(f'\n\n{df}')

    model_path = args.model / 'model.joblib'
    logger.info(f'Loading model {model_path}')
    model = joblib.load(model_path)
    logger.debug(f'Model loaded: {model}')

    X = df.drop(columns=['Label'])
    results = df[['Label']]
    results['Prediction'] = model.predict(X)
    logger.debug(f'\n\n{results}')

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f'Saving predictions to {args.output}')
        results.to_json(args.output, force_ascii=False, indent=4)
        logger.info('Predictions saved')
