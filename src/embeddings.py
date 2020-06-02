"""Train non-contextual embeddings."""

import glob
import logging
import os
import warnings

import fire
from gensim.models import FastText
from gensim.models.callbacks import CallbackAny2Vec
from tqdm import tqdm

from utils import ensure_dir, make_data_dirname

# silence gensim's logging
logging.getLogger("gensim").setLevel(logging.ERROR)
warnings.simplefilter(action="ignore", category=UserWarning)

LOG_DIR = "logs"


class Corpus:
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as file:
            for line in file:
                yield line.strip().split(" ")

    def count(self):
        """Return the number of lines in the corpus."""
        num_lines = 0
        for line in tqdm(self):
            num_lines += 1
        return num_lines


# pylint: disable=W1203
class ProgressCallback(CallbackAny2Vec):
    """Simple callback to display progress during training."""

    def __init__(self, level, lg, epochs):
        super().__init__()
        self.level = level
        self.lg = lg
        self.epochs = epochs
        self.i = 1
        logging.basicConfig(
            filename=os.path.join(LOG_DIR, "embeddings.log"),
            format="%(asctime)s : %(levelname)s : %(message)s",
            datefmt="%d %B %H:%M:%S",
            level=logging.INFO,
        )

    def on_epoch_end(self, model):
        logging.info(f"{self.level} {self.lg} {self.i}/{self.epochs}")
        self.i += 1

    def on_train_begin(self, model):
        logging.info(f"{self.level} {self.lg} Beginning")

    def on_train_end(self, model):
        logging.info(f"{self.level} {self.lg} Finished")


# pylint: disable=W1203
class SaveCallback(CallbackAny2Vec):
    """Simple callback to save model at end of each epoch."""

    def __init__(self, level, lg, name, size, window):
        super().__init__()
        self.i = 1
        self.dirname = f"models/{level}/{lg}/{name}/{size}-{window}"
        ensure_dir(self.dirname)

    def on_epoch_end(self, model):
        filename = os.path.join(self.dirname, f"{self.i}.txt")
        save(model, filename)
        self.i += 1


def save(model, filename):
    model.wv.save_word2vec_format(filename, binary=False)
    # Remove header line
    with open(filename, "r") as file:
        lines = file.readlines()
    with open(filename, "w") as file:
        file.write("".join(lines[1:]))


# pylint: disable=C0330
def train(
    level,  # phoneme or word
    lg,  # language to train on
    size=300,  # size of the embeddings
    window=3,  # context window
    epochs=10,  # number of iterations over the corpus
    min_ngram=3,  # minimum length of n-grams
    max_ngram=6,  # maximum length of n-grams
    min_count=5,  # minimum token frequency
    skipgram=1,  # use skipgram over CBOW
    ngrams=1,  # use fasttext over word2vec
    workers=4,  # number of threads
    save_every_epoch=True,  # whether to save embeddings at every epoch
):
    """Training embeddings.

    Hyperparameters can be specified either at the command line.

    """
    if level == "phoneme":  # force word2vec for phoneme embeddings
        ngrams = 0
    dirname = make_data_dirname(level, lg)
    filename = os.path.join(dirname, "train.txt")
    corpus = Corpus(filename)
    model = FastText(
        size=size,
        window=window,
        min_n=min_ngram,
        max_n=max_ngram,
        min_count=min_count,
        sg=skipgram,
        word_ngrams=ngrams,
        workers=workers,
    )
    model.build_vocab(sentences=corpus)
    callbacks = [ProgressCallback(level, lg, epochs)]
    name = "fasttext" if ngrams else "word2vec"
    if save_every_epoch:
        callbacks.append(SaveCallback(level, lg, name, size, window))
    num_lines = corpus.count()
    model.train(
        sentences=corpus, total_examples=num_lines, epochs=epochs, callbacks=callbacks
    )
    if not save_every_epoch:
        filename = f"models/{level}/{lg}/{name}/{size}-{window}/{epochs}.txt"
        save(model, filename)


def train_phonemes():
    """Train all phoneme-level embedding models"""
    from wikipron import LANGUAGES

    for lg in LANGUAGES.index:
        for size in [5, 10, 20, 30]:
            for window in [1, 2, 3]:
                train("phoneme", lg, size, window, epochs=10, ngrams=0, min_count=1)


if __name__ == "__main__":
    fire.Fire(train)
