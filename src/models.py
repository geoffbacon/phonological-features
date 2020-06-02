"""Tools for interacting with trained models."""
import os

import pandas as pd
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor

from wikipron import END_CHARACTER, START_CHARACTER


def maybe_remove_start_end_chars(embeddings):
    is_start_end_chars = embeddings.index.isin([START_CHARACTER, END_CHARACTER])
    return embeddings[~is_start_end_chars]


def load_static_embeddings(level, lg, name, size, window, epoch):
    filename = f"models/{level}/{lg}/{name}/{size}-{window}/{epoch}.txt"
    embeddings = pd.read_csv(
        filename, delimiter=" ", index_col=0, header=None, names=range(size)
    )
    return maybe_remove_start_end_chars(embeddings)


def load_rnn_model(level, lg, name, size, hidden, epoch="best"):
    serialization_dir = f"models/{level}/{lg}/{name}/{size}-{hidden}"
    if epoch == "best":
        weights_filename = "best.th"
    else:
        weights_filename = f"model_state_epoch_{epoch}.th"
    weights_filename = os.path.join(serialization_dir, weights_filename)
    archive_filename = os.path.join(serialization_dir, "model.tar.gz")
    archive = load_archive(
        archive_filename, cuda_device=-1, weights_file=weights_filename
    )
    return Predictor.from_archive(archive, predictor_name="next_token_lm")


def load_rnn_embeddings(level, lg, name, size, hidden, epoch="best"):
    model = load_rnn_model(level, lg, name, size, hidden, epoch="best")
    params = list(model._model.parameters())
    tensor = params[0]
    vocab = model._model.vocab
    vocab_size = vocab.get_vocab_size()
    assert vocab_size == tensor.shape[0], "Something went wrong!"
    index_to_phoneme = vocab.get_token_from_index
    phonemes = [index_to_phoneme(i) for i in range(vocab_size)]
    embeddings = pd.DataFrame(tensor.data.numpy(), index=phonemes)
    embeddings.drop(["@@UNKNOWN@@", "@@PADDING@@"], axis=0, inplace=True)
    return maybe_remove_start_end_chars(embeddings)


if True:
    level = "phoneme"
    lg = "acw"
    name = "word2vec"
    size = 10
    window = 1
    epoch = 10
else:
    level = "phoneme"
    lg = "fre"
    name = "lstm"
    size = 5
    hidden = 10
    epoch = "best"
