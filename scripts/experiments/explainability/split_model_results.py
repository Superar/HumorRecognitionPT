import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser('Script to split results of a model\'s ' +
                                 'prediction according to whether they are ' +
                                 'correct of not.')
parser.add_argument('data',
                    help='Test data used to obtain the model predictions.',
                    type=Path)
parser.add_argument('predictions',
                    help='Model predictions.',
                    type=Path)
parser.add_argument('output',
                    help='Directory path to save the split results.',
                    type=Path)
args = parser.parse_args()

data = pd.read_json(args.data)
predictions = pd.read_json(args.predictions)

correct = predictions.query('Prediction == Label')
incorrect = predictions.query('Prediction != Label')

args.output.mkdir(parents=True, exist_ok=True)
correct.join(data['Text']).to_json(args.output / 'correct.json',
                                   orient='records',
                                   force_ascii=False,
                                   indent=4)
incorrect.join(data['Text']).to_json(args.output / 'incorrect.json',
                                     orient='records',
                                     force_ascii=False,
                                     indent=4)
