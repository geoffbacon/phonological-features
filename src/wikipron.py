"""Preprocess and prepare Wikipron data."""

import glob
import json
import os
import warnings

import numpy as np
import pandas as pd
from panphon.distance import Distance
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from utils import write

warnings.simplefilter("ignore")

LANGUAGES = pd.read_csv("data/phoneme/languages.csv", index_col=0)
LANGUAGES = LANGUAGES[LANGUAGES["phoible_id"] != -1]
SEED = sum(map(ord, "phonology"))
START_CHARACTER = "<"
END_CHARACTER = ">"

distance_fn = Distance().weighted_feature_edit_distance


def load_wikipron(language):
    filename = f"data/phoneme/wikipron/{language}/raw.tsv"
    return pd.read_csv(
        filename, sep="\t", header=None, names=["orthography", "phonemes", "frequency"]
    )


def wikipron_to_phoible(language):
    phoible_id = LANGUAGES.loc[language, "phoible_id"]
    assert phoible_id != -1, f"No inventory for {language}."
    return phoible_id


def load_inventory(language):
    filename = "data/phoneme/phoible/phoible.csv"
    phoible = pd.read_csv(filename, low_memory=False)
    phoible_id = wikipron_to_phoible(language)
    inventory = phoible[phoible["InventoryID"] == phoible_id]
    no_tones = inventory[inventory["tone"] != "+"]
    no_tones["Phoneme"] = no_tones["Phoneme"].str.replace("ɚ", "ə˞")
    return no_tones


class Preprocessor:
    def __init__(self, language):
        self.language = language
        self.inventory = load_inventory(language)
        self.vocab = self._get_vocab()
        self.table = self._get_table()

    def _get_vocab(self):
        wikipron = load_wikipron(self.language)
        return set(" ".join(wikipron["phonemes"].values).split(" "))

    def _closest(self, char):
        try:
            allophones = self.inventory["Allophones"].str.split()
            is_allophone = allophones.apply(lambda row: char in row)
            num_allophones = is_allophone.sum()
        except TypeError:
            num_allophones = 0
        if num_allophones == 0:
            return min(self.inventory["Phoneme"], key=lambda ph: distance_fn(ph, char))
        elif num_allophones == 1:
            return self.inventory[is_allophone]["Phoneme"].iloc[0]
        else:
            return min(
                self.inventory[is_allophone]["Phoneme"],
                key=lambda ph: distance_fn(ph, char),
            )

    def _get_table(self):
        not_in_inventory = self.vocab.difference(set(self.inventory["Phoneme"]))
        table = {}
        for char in not_in_inventory:
            table[char] = self._closest(char)
        return table

    def preprocess(self, form):
        chars = form.split(" ")
        phonemes = [self.table.get(ch, ch) for ch in chars]
        phonemes = [START_CHARACTER] + phonemes + [END_CHARACTER]
        return " ".join(phonemes)


def _prepare(language):
    wikipron = load_wikipron(language)
    # Preprocess
    preprocessor = Preprocessor(language)
    wikipron["preprocessed"] = wikipron["phonemes"].apply(preprocessor.preprocess)
    # Filter
    wikipron = wikipron[
        wikipron["preprocessed"].str.split(" ").apply(len) > 3
    ]  # at least two phonemes
    # Split
    train, valid = train_test_split(
        wikipron, train_size=0.9, random_state=SEED, shuffle=True
    )
    # Order train
    train = train.sample(frac=1)
    train["length"] = train["preprocessed"].str.len()
    train.sort_values(by=["frequency", "length"], ascending=(False, True), inplace=True)
    # Write
    filename = f"data/phoneme/wikipron/{language}/train.txt"
    write(train["preprocessed"], filename)
    filename = f"data/phoneme/wikipron/{language}/validation.txt"
    write(valid["preprocessed"], filename)


def prepare():
    for lg in tqdm(LANGUAGES.index):
        _prepare(lg)

def count(lg):
    """Count the number of editsmy preprocessing performs."""
    wikipron = load_wikipron(lg)
    preprocessor = Preprocessor(lg)
    inventory = load_inventory(lg)
    i = set(inventory["Phoneme"])
    result = []
    for form in wikipron["phonemes"]:
        for ch in form.split(" "):
            if ch not in i:
                result.append({"form": form, "ch": ch})
                print(f"{lg} | {form} | {ch}")
    result = pd.DataFrame(result)



if __name__ == "__main__":
    prepare()
