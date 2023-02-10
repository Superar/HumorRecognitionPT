from argparse import ArgumentParser
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

parser = ArgumentParser('Script to calculate feature importance in a given model ' +
                        'which is not callable for a given dataset using SHAP')
parser.add_argument('train', type=Path,
                    help='Data in HDF5 used to train the model.')
parser.add_argument('test', type=Path,
                    help='Data in HDF5 to run the tests with.')
parser.add_argument('model', type=Path,
                    help='Path to the directory containing the trained model.')
parser.add_argument('output', type=Path,
                    help='Output directory name to save SHAP data and plots.')
parser.add_argument('--max_display',
                    help='Maximum number of features to take into account.',
                    required=False, type=int,
                    default=10)
args = parser.parse_args()

print(f'Loading file {args.train}')
df_train = pd.read_hdf(args.train)
X_train = df_train.drop(columns=['Label'])

print(f'Loading file {args.test}')
df_test = pd.read_hdf(args.test)
X_test = df_test.drop(columns=['Label'])

model_path = args.model / 'model.joblib'
print(f'Loading model {model_path}')
model = joblib.load(model_path)
humor_index = np.where(model.classes_ == 'H')[0][0]
print(f'Model loaded: {model}')

med = X_train.median().values.reshape((1, X_train.shape[1]))
explainer = shap.KernelExplainer(model.predict_proba, med)
shap_values = explainer.shap_values(X_test)
shap_explanation = shap.Explanation(shap_values[humor_index],
                                    base_values=explainer.expected_value[humor_index],
                                    data=X_test.values,
                                    feature_names=X_test.columns)

args.output.mkdir(parents=True, exist_ok=True)
shap.plots.beeswarm(shap_explanation,
                    max_display=args.max_display,
                    show=False)
plt.tight_layout()
plt.savefig(args.output / 'beeswarm_H.pdf')

plt.clf()
shap.plots.bar(shap_explanation,
               max_display=args.max_display,
               show=False)
plt.tight_layout()
plt.savefig(args.output / 'bar_H.pdf')

plt.clf()
shap.plots.bar(shap_explanation,
               max_display=shap_values[humor_index].shape[1],
               show=False)
plt.tight_layout()
plt.savefig(args.output / 'bar_H_max.pdf')

plt.clf()
args.output.mkdir(parents=True, exist_ok=True)
shap.plots.beeswarm(shap_explanation,
                    max_display=shap_values[humor_index].shape[1],
                    show=False)
plt.tight_layout()
plt.savefig(args.output / 'beeswarm_H_max.pdf')
