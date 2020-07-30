"""Correlation of dissimilarity."""
import os

import numpy as np
import pandas as pd
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr
from tqdm import tqdm

from models import embeddings_to_dissimilarity, load_embeddings
from utils import ensure_dir
from wikipron import distance_fn, load_features


def correlate(**kwargs):
    # Get pairwise dissimilarities of learnt representations
    embeddings = load_embeddings(**kwargs)
    learnt_dissimilarity_matrix = embeddings_to_dissimilarity(embeddings)
    learnt_dissimilarities = squareform(learnt_dissimilarity_matrix)
    # Get pairwise dissimilarities of ground truth representations
    level, lg = kwargs["level"], kwargs["lg"]
    assert level == "phoneme", "This function is only for phoneme-level models"
    phonemes_in_corpus = list(learnt_dissimilarity_matrix.index)
    features = load_features(lg).loc[phonemes_in_corpus]
    row_indices, column_indices = np.triu_indices(len(learnt_dissimilarity_matrix), k=1)
    rows = [learnt_dissimilarity_matrix.index[i] for i in row_indices]
    columns = [learnt_dissimilarity_matrix.columns[j] for j in column_indices]
    indices = list(zip(rows, columns))
    ground_truth_dissimilarities = np.array([distance_fn(*index) for index in indices])
    r, p = spearmanr(learnt_dissimilarities, ground_truth_dissimilarities)
    # Write results to disk
    if "hidden" in kwargs:
        hyperparams = f"{kwargs['size']}-{kwargs['hidden']}"
    else:
        hyperparams = f"{kwargs['size']}-{kwargs['window']}"
    name = kwargs["name"]
    path = f"results/{level}/correlation/{lg}/{name}/{hyperparams}"
    ensure_dir(path)
    epoch = kwargs["epoch"]
    filename = os.path.join(path, f"{epoch}.txt")
    with open(filename, "w") as file:
        file.write(str((r, p)))
    return r, p


def get_raw_dissimilarities(**kwargs):
    # Get pairwise dissimilarities of learnt representations
    embeddings = load_embeddings(**kwargs)
    learnt_dissimilarity_matrix = embeddings_to_dissimilarity(embeddings)
    learnt_dissimilarities = squareform(learnt_dissimilarity_matrix)
    # Get pairwise dissimilarities of ground truth representations
    level, lg = kwargs["level"], kwargs["lg"]
    assert level == "phoneme", "This function is only for phoneme-level models"
    phonemes_in_corpus = list(learnt_dissimilarity_matrix.index)
    features = load_features(lg).loc[phonemes_in_corpus]
    row_indices, column_indices = np.triu_indices(len(learnt_dissimilarity_matrix), k=1)
    rows = [learnt_dissimilarity_matrix.index[i] for i in row_indices]
    columns = [learnt_dissimilarity_matrix.columns[j] for j in column_indices]
    indices = list(zip(rows, columns))
    ground_truth_dissimilarities = np.array([distance_fn(*index) for index in indices])
    return pd.DataFrame(
        {"learnt": learnt_dissimilarities, "true": ground_truth_dissimilarities},
        index=indices,
    )


def main():
    from models import all_trained_phoneme_models

    models = all_trained_phoneme_models()
    for kwargs in tqdm(models):
        size = kwargs["size"]
        if size != "groundTruth":
            _ = correlate(**kwargs)


if __name__ == "__main__":
    main()
