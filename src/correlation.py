"""Correlation of dissimilarity."""

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from models import embeddings_to_dissimilarity, load_embeddings


def correlate(embeddings, ground_truth):
    # Pairwise dissimilarities of learnt representations
    learnt_dissimilarity_matrix = embeddings_to_dissimilarity(embeddings)
    learnt_dissimilarities = squareform(learnt_dissimilarity_matrix)

    # Pairwise dissimilarities of ground truth representations
    objects = list(learnt_dissimilarity_matrix.index)
    row_indices, column_indices = np.triu_indices(len(learnt_dissimilarity_matrix), k=1)
    rows = [learnt_dissimilarity_matrix.index[i] for i in row_indices]
    columns = [learnt_dissimilarity_matrix.columns[j] for j in column_indices]
    indices = list(zip(rows, columns))
