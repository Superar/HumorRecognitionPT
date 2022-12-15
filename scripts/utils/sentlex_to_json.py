from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

parser = ArgumentParser()
parser.add_argument('input', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()

polarity = dict()
with args.input.open(encoding='utf-8') as file_:
    for line in file_:
        line = line.split('.')
        words = line[0].split(',')

        # Not considering expressions
        for word in words:
            if ' ' not in word:
                value = line[1].split(';')
                value = value[3].split('=')

                polarity[word] = value[1]

df = pd.DataFrame.from_dict(polarity, orient='index',
                            columns=['Polarity'])
df.to_json(args.output, orient='index',
           force_ascii=False, indent=4)
