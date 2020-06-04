"""Correlation with ground truth representations."""
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import linalg
from scipy.linalg import decomp_qr
from scipy.stats import pearsonr
from sklearn import preprocessing
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
    level, name = kwargs["level"], kwargs["name"]
    if "hidden" in kwargs:
        hyperparams = f"{kwargs['size']}-{kwargs['hidden']}"
    else:
        hyperparams = f"{kwargs['size']}-{kwargs['window']}"
    path = f"results/{level}/qvec/{lg}/{name}/{hyperparams}"
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
    level, lg, name = kwargs["level"], kwargs["lg"], kwargs["name"]
    if "hidden" in kwargs:
        hyperparams = f"{kwargs['size']}-{kwargs['hidden']}"
    else:
        hyperparams = f"{kwargs['size']}-{kwargs['window']}"
    path = f"results/{level}/qvec/{lg}/{name}/{hyperparams}"
    ensure_dir(path)
    epoch = kwargs["epoch"]
    filename = os.path.join(path, f"{epoch}.png")
    fig = plt.gcf()
    fig.savefig(filename)


def qvec_cca(**kwargs):
    embeddings = load_embeddings(**kwargs).T
    lg = kwargs["lg"]
    features = load_features(lg).T
    common_phonemes = embeddings.columns.intersection(features.columns)
    S = features[common_phonemes]
    X = embeddings[common_phonemes]
    cca = CCA(n_components=1)
    a, b = cca.fit_transform(X.T, S.T)
    a, b = a.reshape(-1), b.reshape(-1)
    r, p = pearsonr(a, b)
    # Write results to disk
    level, lg, name = kwargs["level"], kwargs["lg"], kwargs["name"]
    if "hidden" in kwargs:
        hyperparams = f"{kwargs['size']}-{kwargs['hidden']}"
    else:
        hyperparams = f"{kwargs['size']}-{kwargs['window']}"
    path = f"results/{level}/qvec/{lg}/{name}/{hyperparams}"
    ensure_dir(path)
    epoch = kwargs["epoch"]
    filename = os.path.join(path, f"{epoch}.txt")
    with open(filename, "w") as file:
        file.write(str((r, p)))
    return r, p


def norm_center_matrix(m):
    m = preprocessing.normalize(m)
    m_mean = m.mean(axis=0)
    m -= m_mean
    return m


def original(**kwargs):
    embeddings = load_embeddings(**kwargs)
    lg = kwargs["lg"]
    features = load_features(lg)
    common_phonemes = embeddings.index.intersection(features.index)
    S = features.loc[common_phonemes]
    X = embeddings.loc[common_phonemes]
    assert X.shape[0] == S.shape[0], (X.shape, S.shape, "Unequal number of rows")
    assert X.shape[0] > 1, (X.shape, "Must have more than 1 row")
    X = norm_center_matrix(X)
    S = norm_center_matrix(S)
    X_q, _, _ = decomp_qr.qr(X, overwrite_a=True, mode="economic", pivoting=True)
    S_q, _, _ = decomp_qr.qr(S, overwrite_a=True, mode="economic", pivoting=True)
    C = np.dot(X_q.T, S_q)
    r = linalg.svd(C, full_matrices=False, compute_uv=False)
    d = min(X.shape[1], S.shape[1])
    r = r[:d]
    r = np.minimum(np.maximum(r, 0.0), 1.0)  # remove roundoff errs
    r = r.mean()
    # Write results to disk
    level, lg, name = kwargs["level"], kwargs["lg"], kwargs["name"]
    if "hidden" in kwargs:
        hyperparams = f"{kwargs['size']}-{kwargs['hidden']}"
    else:
        hyperparams = f"{kwargs['size']}-{kwargs['window']}"
    path = f"results/{level}/qvec/{lg}/{name}/{hyperparams}"
    ensure_dir(path)
    epoch = kwargs["epoch"]
    filename = os.path.join(path, f"{epoch}.txt")
    with open(filename, "w") as file:
        file.write(str(r))
    return r


def main():
    from models import all_trained_phoneme_models

    models = all_trained_phoneme_models()
    for kwargs in tqdm(models):
        size = kwargs["size"]
        if size != "groundTruth":
            _ = heatmap(**kwargs)
            _ = original(**kwargs)


if __name__ == "__main__":
    main()
