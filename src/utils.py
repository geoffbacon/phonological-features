"""Utilities used across multiple modules."""

import os
from multiprocessing import Pool


def parallelize(func, data, n_workers=4):
    num_data = len(data)
    chunk_size = (num_data // n_workers) + 1
    chunked_data = [data[i : i + chunk_size] for i in range(0, num_data, chunk_size)]
    with Pool(n_workers) as pool:
        pool.map(func, chunked_data)


def ensure_dir(dirname):
    """Ensure the path to `dirname` exists, creating it if it doesn't."""
    os.makedirs(dirname, exist_ok=True)


def write(lines, filename):
    with open(filename, "w") as file:
        file.write("\n".join(lines))


def make_data_dirname(level, lg):
    dataset = "wikipron" if level == "phoneme" else "wiki40b"
    return f"data/{level}/{dataset}/{lg}"
