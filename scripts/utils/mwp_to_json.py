from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument('input', type=Path)
parser.add_argument('output')
args = parser.parse_args()

df = pd.read_excel(args.input, sheet_name='MWP norms')
df.to_json(args.output, force_ascii=False)
