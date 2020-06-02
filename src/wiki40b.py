"""Download, preprocess and prepare Wiki40B data."""
import os
import re

import apache_beam as beam
import fire
import stanza
import tensorflow_datasets as tfds
from tqdm import tqdm

from utils import parallelize, write

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # silence TensorFlow warnings

PARAGRAPH_PATTERN = re.compile(r"_START_PARAGRAPH_\n(.*)")
NEWLINE_PATTERN = re.compile(r"_NEWLINE_")
BATCH_SIZE = 32
MAX_SEQLEN = 1000

# Commented out languages do not have a corresponding tokenizer in Stanza
LANGUAGES = [
    "ar",
    "bg",
    "ca",
    "cs",
    "da",
    "de",
    "el",
    "en",
    "es",
    "et",
    "fa",
    "fi",
    "fr",
    "he",
    "hi",
    "hr",
    "hu",
    "id",
    "it",
    "ja",
    "ko",
    "lt",
    "lv",
    # "ms",
    "nl",
    "no",
    "pl",
    "pt",
    "ro",
    "ru",
    "sk",
    "sl",
    "sr",
    "sv",
    # "th",
    # "tl",
    "tr",
    "uk",
    "vi",
    "zh-cn",
    "zh-tw",
]


def download(lg):
    """Download the Wiki40b data for `lg`, unless it is already downloaded on this machine."""
    options = beam.options.pipeline_options.PipelineOptions()
    config = tfds.download.DownloadConfig(beam_options=options)
    ds = tfds.load(
        f"wiki40b/Wiki40B.{lg}", download_and_prepare_kwargs={"download_config": config}
    )


def download_all():
    for lg in LANGUAGES:
        download(lg)


def lines_from_article(article):
    """Return list of strings representing all lines in `article`."""
    text = article["text"].numpy().decode("utf-8")
    paragraph_texts = PARAGRAPH_PATTERN.findall(text)
    lines = []
    for paragraph_text in paragraph_texts:
        lines.extend(NEWLINE_PATTERN.split(paragraph_text))
    return lines


def get_stanza_tokenizer(lg):
    """Return stanza.Pipeline for `lg` with just the tokenizer.

    This will download the stanza model if it's not already downloaded.
    Stanza uses different language codes for Chinese, so we take care
    of that here.
    """
    if lg == "zh-cn":
        stanza_lg = "zh-hans"
    elif lg == "zh-tw":
        stanza_lg = "zh-hant"
    else:
        stanza_lg = lg
    stanza.download(stanza_lg, processors="tokenize", verbose=False)
    config = {
        "lang": stanza_lg,
        "processors": "tokenize",
        "tokenize_batch_size": BATCH_SIZE,
        "tokenize_max_seqlen": MAX_SEQLEN,
        "verbose": False,
    }
    return stanza.Pipeline(**config)


def _prepare(lg, debug=True):
    download(lg)
    ds = tfds.load(f"wiki40b/Wiki40B.{lg}")
    nlp = get_stanza_tokenizer(lg)
    for split in ["train", "validation", "test"]:
        data = ds[split]
        if debug:
            data = list(data.take(100))
        raw_lines = []
        for article in tqdm(data):
            raw_lines.extend(lines_from_article(article))
        num_lines = len(raw_lines)
        chunk_size = BATCH_SIZE * 4
        processed_lines = []
        for i in tqdm(range(0, num_lines, chunk_size)):
            chunk = raw_lines[i : i + chunk_size]
            doc = nlp("\n\n".join(chunk))
            sentences = [
                " ".join([token.text for token in sentence.tokens])
                for sentence in doc.sentences
            ]
            processed_lines.extend(sentences)
        filename = f"data/word/wiki40b/{lg}/{split}.txt"
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        write(processed_lines, filename)


def prepare():
    # TODO equally distribute larger languages
    parallelize(_prepare, LANGUAGES, n_workers=4)


if __name__ == "__main__":
    fire.Fire(_prepare)
