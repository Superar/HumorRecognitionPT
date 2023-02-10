from argparse import ArgumentParser
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

parser = ArgumentParser('Script to calculate feature importance in a given callable ' +
                        'model for a given dataset using SHAP')
parser.add_argument('input', type=Path,
                    help='Input data in HDF5 to run the tests with.')
parser.add_argument('model', type=Path,
                    help='Path to the directory containing the trained model.')
parser.add_argument('output', type=Path,
                    help='Output directory name to save SHAP data and plots.')
parser.add_argument('--max_display',
                    help='Maximum number of features to take into account.',
                    required=False, type=int,
                    default=10)
args = parser.parse_args()

print(f'Loading file {args.input}')
df = pd.read_hdf(args.input)
X = df.drop(columns=['Label'])

model_path = args.model / 'model.joblib'
print(f'Loading model {model_path}')
model = joblib.load(model_path)
humor_index = np.where(model.classes_ == 'H')[0][0]
print(f'Model loaded: {model}')

explainer = shap.Explainer(model)
shap_values = explainer(X)

args.output.mkdir(parents=True, exist_ok=True)
shap.plots.beeswarm(shap_values[:, :, humor_index],
                    max_display=args.max_display,
                    show=False)
plt.tight_layout()
plt.savefig(args.output / 'beeswarm_H.pdf')

plt.clf()
shap.plots.bar(shap_values[:, :, humor_index],
               max_display=args.max_display,
               show=False)
plt.tight_layout()
plt.savefig(args.output / 'bar_H.pdf')

plt.clf()
shap.plots.bar(shap_values[:, :, humor_index],
               max_display=shap_values.shape[0],
               show=False)
plt.tight_layout()
plt.savefig(args.output / 'bar_H_max.pdf')

plt.clf()
args.output.mkdir(parents=True, exist_ok=True)
shap.plots.beeswarm(shap_values[:, :, humor_index],
                    max_display=shap_values.shape[0],
                    show=False)
plt.tight_layout()
plt.savefig(args.output / 'beeswarm_H_max.pdf')