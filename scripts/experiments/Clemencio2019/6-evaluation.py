from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from sklearn.metrics import classification_report

parser = ArgumentParser()
parser.add_argument('input',
                    help='Path to the directory containing all prediction JSON files',
                    type=Path)
args = parser.parse_args()

dfs = dict()
for path in args.input.iterdir():
    results = pd.read_json(path)
    evaluation = classification_report(results['Label'],
                                    results['Prediction'],
                                    output_dict=True)
    evaluation = pd.DataFrame.from_dict(evaluation)
    dfs[path.stem] = evaluation

evaluation = pd.concat(dfs, names=['Method'])
print(evaluation)
