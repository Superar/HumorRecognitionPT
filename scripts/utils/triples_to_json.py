import argparse
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('input', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--relations', '-r', nargs='*')
parser.add_argument('--min_sources', '-m',
                    type=int, required=False,
                    default=1)
args = parser.parse_args()

df = pd.read_csv(args.input, sep='\t',
                 names=['Triple', 'Number of sources'])
triples = pd.DataFrame(df['Triple'].str.split().to_list())
df[['Argument 1', 'Relation', 'Argument 2']] = triples

if args.relations:
    print(f'Keeping relations: {args.relations}\n')
    drop_rels = pd.Index(df["Relation"]).difference(args.relations).unique()
    print(f'Dropping relations: {drop_rels.to_list()}\n')
    df = df.loc[df['Relation'].isin(args.relations), :]

print(f'Minimum number of sources: {args.min_sources}')
df = df.loc[df['Number of sources'] >= args.min_sources]

print(f'Saving file to {args.output}')
df.to_json(args.output, force_ascii=False,
           indent=4, orient='records')
