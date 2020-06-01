"""Train RNN-based language models."""

import itertools
import json
import os
import re

import _jsonnet
import fire
import torch

CONFIG_FILENAME = "src/config.jsonnet"
TMP_CONFIG_FILENAME = "src/tmp.jsonnet"
TRAIN_CMD = "allennlp train -s {directory} -f {config} && rm {config}"
GPU_AVAILABLE = str(torch.cuda.is_available()).lower()


def prepare_config(options):
    with open(CONFIG_FILENAME) as file:
        config = file.read()
    for key, value in options.items():
        pattern = f'local {key} = ["a-z0-9]+;'
        if isinstance(value, str) and value not in ["true", "false"]:
            repl = f"local {key} = '{value}';"
        else:
            repl = f"local {key} = {value};"
        config = re.sub(pattern, repl, config)
    return config


def train(level, lg, name, size, hidden, epochs=10):
    options = {
        "LEVEL": level,
        "LANGUAGE": lg,
        "NAME": name,
        "SIZE": size,
        "HIDDEN": hidden,
        "NUM_EPOCHS": epochs,
        "USE_GPU": GPU_AVAILABLE,
        "BATCH_SIZE": 16,
    }
    config_str = prepare_config(options)
    config = json.loads(_jsonnet.evaluate_snippet("snippet", config_str))
    # The override flag in allennlp was finicky so I used a temporary file hack
    with open(TMP_CONFIG_FILENAME, "w") as file:
        json.dump(config, file, indent=2)
    serialization_dir = f"models/{level}/{lg}/{name}/{size}-{hidden}"
    cmd = TRAIN_CMD.format(directory=serialization_dir, config=TMP_CONFIG_FILENAME)
    os.system(cmd)


def sweep(level, lg, name):
    SIZES = [20, 40, 60, 80, 100, "groundTruth"]
    HIDDENS = [50, 100, 200]
    for hyperparams in itertools.product(SIZES, HIDDENS):
        size = hyperparams[0]
        hidden = hyperparams[1]
        train(level=level, lg=lg, name=name, size=size, hidden=hidden)


if __name__ == "__main__":
    fire.Fire(train)
