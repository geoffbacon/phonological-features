"""Script to make ground truth phonological feature representations.

$ python src/features.py
"""

import os

import fire
import pandas as pd
import panphon
from tqdm import tqdm

from utils import write
from wikipron import LANGUAGES, load_inventory

FEATURES = panphon.FeatureTable()


def feature_vector(phoneme):
    return FEATURES.word_fts(phoneme)[0].numeric()


def create_features(language):
    inventory = load_inventory(language)
    phonemes = inventory["Phoneme"]
    representations = {phoneme: feature_vector(phoneme) for phoneme in phonemes}
    features = pd.DataFrame(representations, index=FEATURES.names).T
    os.makedirs(f"data/phoneme/features/{language}", exist_ok=True)
    filename = f"data/phoneme/features/{language}/features.csv"
    features.to_csv(filename)
    features_as_txt = format_to_txt(features)
    filename = f"data/phoneme/features/{language}/features.txt"
    write(features_as_txt, filename)


def format_to_txt(df):
    result = []
    for phoneme, row in df.iterrows():
        values = " ".join(map(str, row.values))
        result.append(phoneme + " " + values)
    return result


def main():
    for lg in tqdm(LANGUAGES.index):
        create_features(lg)


if __name__ == "__main__":
    main()
