"""Script to make analogy data.

$ python src/analogy.py
"""
import os

import pandas as pd
from tqdm import tqdm

from models import load_keyedvectors
from utils import ensure_dir
from wikipron import load_features


def make_analogies(lg):
    features = load_features(lg)
    inventory_size = len(features)
    minimal_pairs = []
    for i in range(inventory_size):
        this_representation = features.iloc[i, :]
        for j in range(i + 1, inventory_size):
            other_representation = features.iloc[j, :]
            differences = ~(this_representation == other_representation)
            num_differences = differences.sum()
            if num_differences == 1:
                a, b = features.index[i], features.index[j]
                ft = differences[differences].index[0]
                minimal_pairs.append((a, b, ft))
    num_minimal_pairs = len(minimal_pairs)
    analogies = []
    for i in range(num_minimal_pairs):
        this_pair = minimal_pairs[i]
        for j in range(i + 1, num_minimal_pairs):
            other_pair = minimal_pairs[j]
            if this_pair[2] == other_pair[2]:  # if they differ by the same feature
                a, b, ft = this_pair
                c, d, _ = other_pair  # _ is same as ft
                analogies.append((a, b, c, d, ft))
    return pd.DataFrame(analogies, columns=["a", "b", "c", "d", "ft"])


def do_one_analogy(row, vectors):
    """Return the index of `d` in the results of `a - b + c`."""
    a, b, c, d, _ = row
    try:
        result = vectors.most_similar(
            positive=[a, c], negative=[b], topn=len(vectors.vocab)
        )
        phonemes = [r[0] for r in result]
        return phonemes.index(d)
    except (KeyError, ValueError):  # one of the four phonemes was not in corpus
        return -1


def score_model(**kwargs):
    vectors = load_keyedvectors(**kwargs)
    analogies = make_analogies(kwargs["lg"])
    analogies["rank"] = analogies.apply(do_one_analogy, vectors=vectors, axis=1)
    # Write results to disk
    level, name, lg = kwargs["level"], kwargs["name"], kwargs["lg"]
    if "hidden" in kwargs:
        hyperparams = f"{kwargs['size']}-{kwargs['hidden']}"
    else:
        hyperparams = f"{kwargs['size']}-{kwargs['window']}"
    path = f"results/{level}/analogy/{lg}/{name}/{hyperparams}"
    ensure_dir(path)
    epoch = kwargs["epoch"]
    filename = os.path.join(path, f"{epoch}.csv")
    analogies.to_csv(filename, index=False)
    return analogies


def main():
    from models import all_trained_phoneme_models

    models = all_trained_phoneme_models()
    for model in tqdm(models):
        if model["size"] != "groundTruth":
            _ = score_model(**model)


if __name__ == "__main__":
    main()
