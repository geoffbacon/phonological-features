"""Correlation with ground truth representations."""
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.cross_decomposition import CCA
from tqdm import tqdm

from models import load_embeddings
from utils import ensure_dir
from wikipron import load_features


def correlate(**kwargs):
    embeddings = load_embeddings(**kwargs).T
    lg = kwargs["lg"]
    features = load_features(lg).T
    common_phonemes = embeddings.columns.intersection(features.columns)
    S = features[common_phonemes]
    X = embeddings[common_phonemes]
    correlations = pd.DataFrame(
        {i: X.corrwith(S.iloc[i], axis=1) for i in range(len(S))}
    )
    correlations.columns = S.index
    # Write results to disk
    level = kwargs["level"]
    if "hidden" in kwargs:
        hyperparams = f"{kwargs['size']}-{kwargs['hidden']}"
    else:
        hyperparams = f"{kwargs['size']}-{kwargs['window']}"
    path = f"results/{level}/qvec/{lg}/{hyperparams}"
    ensure_dir(path)
    epoch = kwargs["epoch"]
    filename = os.path.join(path, f"{epoch}.csv")
    correlations.to_csv(filename, index=False)
    return correlations


def heatmap(**kwargs):
    correlations = correlate(**kwargs).abs()
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(correlations, cmap="OrRd", vmin=0, vmax=1, annot=False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    # fix matplotlib regression in 3.1.1
    b, t = plt.ylim()
    b += 0.5
    t -= 0.5
    plt.ylim(b, t)
    locs, labels = plt.yticks()
    plt.setp(labels, rotation=0)
    # Write results to disk
    level, lg = kwargs["level"], kwargs["lg"]
    if "hidden" in kwargs:
        hyperparams = f"{kwargs['size']}-{kwargs['hidden']}"
    else:
        hyperparams = f"{kwargs['size']}-{kwargs['window']}"
    path = f"results/{level}/qvec/{lg}/{hyperparams}"
    ensure_dir(path)
    epoch = kwargs["epoch"]
    filename = os.path.join(path, f"{epoch}.png")
    ensure_dir(filename)
    fig = plt.gcf()
    fig.savefig(filename)


def qvec_cca(language, char, hidden, epoch):
    embeddings = load_embeddings(language, char, hidden, epoch).T
    features = load_features(language).T
    common_phonemes = embeddings.columns.intersection(features.columns)
    S = features[common_phonemes]
    X = embeddings[common_phonemes]
    cca = CCA(n_components=1)
    a, b = cca.fit_transform(X.T, S.T)
    a, b = a.reshape(-1), b.reshape(-1)
    r, p = pearsonr(a, b)
    # Write results to disk
    level, lg = kwargs["level"], kwargs["lg"]
    if "hidden" in kwargs:
        hyperparams = f"{kwargs['size']}-{kwargs['hidden']}"
    else:
        hyperparams = f"{kwargs['size']}-{kwargs['window']}"
    path = f"results/{level}/qvec/{lg}/{hyperparams}"
    ensure_dir(path)
    epoch = kwargs["epoch"]
    filename = os.path.join(path, f"{epoch}.txt")
    with open(filename, "w") as file:
        file.write(str((r, p)))
    return r, p


def main():
    from models import all_trained_phoneme_models

    models = all_trained_phoneme_models()
    for kwargs in tqdm(models):
        size = kwargs["size"]
        if size != "groundTruth":
            _ = heatmap(**kwargs)
            _ = qvec_cca(**kwargs)


if __name__ == "__main__":
    main()
