import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument('input')
parser.add_argument('output', type=Path)
parser.add_argument('--ratio', default=0.3, type=float)
args = parser.parse_args()

df = pd.read_hdf(args.input)
train, test = train_test_split(df, test_size=args.ratio)
print(f'Train\n\n{train}')
print(f'Test\n\n{test}')

args.output.mkdir(parents=True, exist_ok=True)
train.to_hdf(args.output / 'train.hdf5', key='train', mode='w')
test.to_hdf(args.output / 'test.hdf5', key='test', mode='w')
