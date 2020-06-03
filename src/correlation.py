"""Correlation of dissimilarity."""

import numpy as np
from scipy.spatial.distance import squareform
from scipy.stats import spearmanr

from models import embeddings_to_dissimilarity, load_embeddings
from wikipron import load_features


def correlate(**kwargs):
    embeddings = load_embeddings(**kwargs)
    dissimilarities = embeddings_to_dissimilarity(embeddings)
    level, lg = kwargs["level"], kwargs["lg"]
    assert level == "phoneme", "This function is only for phoneme-level models"
    features = load_features(lg)
