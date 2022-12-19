import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('input', type=Path)
parser.add_argument('output')
args = parser.parse_args()

items = list()
with args.input.open('r', encoding='utf-8') as file_:
    for line in file_:
        items.append(line.strip())
series = pd.Series(items)
series.to_json(args.output, force_ascii=False, indent=4)
