import pandas as pd
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('input')
parser.add_argument('output')
args = parser.parse_args()

df = pd.read_csv(args.input, sep='\t',
                 names=['Text', 'Label'])
df.to_json(args.output, orient='records',
           force_ascii=False, indent=4)
