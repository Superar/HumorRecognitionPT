import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance

parser = argparse.ArgumentParser()
parser.add_argument('data',
                    help='Data in HDF5 format.',
                    type=Path)
parser.add_argument('model',
                    help='Model directory path.',
                    type=Path)
parser.add_argument('output',
                    help='Output file with results.',
                    type=Path)
args = parser.parse_args()

df = pd.read_hdf(args.data)
X = df.drop(columns=['Label'])
y = df[['Label']]

model_path = args.model / 'model.joblib'
model = joblib.load(model_path)

importance = permutation_importance(model, X, y,
                                    n_repeats=500,
                                    n_jobs=-1,
                                    random_state=0)
importance_df = pd.DataFrame(importance.importances,
                             index=X.columns)
importance_df.to_excel(args.output, engine='odf')
