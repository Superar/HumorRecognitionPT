import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
from scipy.spatial.distance import cdist
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans


def get_centroids(data, centers):
    dist_mat = distance_matrix(centers, np.asmatrix(data))
    centroids = np.argmin(dist_mat, axis=1)
    return centroids


parser = argparse.ArgumentParser('Script to perform clustering on sentences' +
                                 'using a transformer sentence embedding model' +
                                 'and K-Means')
parser.add_argument('--input', '-i',
                    help='Data to be clustered in JSON format',
                    required=True, type=Path)
parser.add_argument('--output', '-o',
                    help='Output file to save the clustering results in JSON format.',
                    required=True, type=Path)
parser.add_argument('--model', '-m',
                    help='HuggingFace model to use',
                    required=False, type=str,
                    default='rufimelo/bert-large-portuguese-cased-sts')
parser.add_argument('--num_clusters', '-n',
                    help='Number of clusters to use',
                    required=False, type=int,
                    default=56)
args = parser.parse_args()

df = pd.read_json(args.input)
model = SentenceTransformer(args.model)

sentence_embeddings = model.encode(df['Text'])
clustering_model = KMeans(n_clusters=args.num_clusters)
clustering_model.fit(sentence_embeddings)
df['Cluster'] = clustering_model.labels_

df['Is Centroid'] = False
centroid_indices = get_centroids(sentence_embeddings,
                                 clustering_model.cluster_centers_)
df.loc[centroid_indices, 'Is Centroid'] = True

df.to_json(args.output, force_ascii=False,
           orient='records', indent=4)
