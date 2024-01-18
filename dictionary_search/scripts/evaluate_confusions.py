import argparse
import csv
import dataclasses
import glob
import json
import os
import random
import time
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
from dictionary_search_model import Model

_DATABASE = []


@dataclasses.dataclass
class DatabaseEntry:
    embedding: np.ndarray
    gloss: str


def populate_db(database_path: str, max_entries: int):
    # We first add the signs that are in the query set and then continue to add unique signs until we reach max_entries.
    # We do this in a deterministic manner of course...
    db_entries = [os.path.join(database_path, 'HEBBEN-A-4801.npy'),  # In corpus.
                  os.path.join(database_path, 'TELEFONEREN-D-11870.npy'),  # In corpus.
                  os.path.join(database_path, 'HAAS-A-16146.npy'),  # In corpus.
                  os.path.join(database_path, 'STRAAT-A-11560.npy'),  # In corpus.
                  os.path.join(database_path, 'PAARD-A-8880.npy'),  # In corpus.
                  os.path.join(database_path, 'HOND-A-5052.npy'),  # In corpus.
                  os.path.join(database_path, 'RUSTEN-B-10250.npy'),  # In corpus.
                  os.path.join(database_path, 'SCHOOL-A-10547.npy'),  # In corpus.
                  os.path.join(database_path, 'ONTHOUDEN-A-8420.npy'),  # In corpus.
                  os.path.join(database_path, 'WAT-A-13657.npy'),  # In corpus.

                  os.path.join(database_path, 'BOUWEN-G-1906.npy'),  # Not in corpus.
                  os.path.join(database_path, 'WAAROM-A-13564.npy'),  # Not in corpus.
                  os.path.join(database_path, 'MELK-B-7418.npy'),  # Not in corpus.
                  os.path.join(database_path, 'VALENTIJN-A-16235.npy'),  # Not in corpus.
                  os.path.join(database_path, 'HERFST-B-4897.npy'),  # Not in corpus.
                  os.path.join(database_path, 'VLIEGTUIG-B-13187.npy'),  # Not in corpus.
                  os.path.join(database_path, 'KLEPELBEL-A-1166.npy'),  # Not in corpus.
                  os.path.join(database_path, 'POES-G-9372.npy'),  # Not in corpus.
                  os.path.join(database_path, 'MOEDER-A-7676.npy'),  # Not in corpus.
                  os.path.join(database_path, 'VADER-G-8975.npy')]  # Not in corpus.

    all_db_entries: List[str] = glob.glob(os.path.join(database_path, '*.npy'))
    indices = list(range(len(all_db_entries)))
    random.seed(10)
    random.shuffle(indices)
    with open('/tmp/indices.txt', 'w') as f:
        f.writelines([str(i) + '\n' for i in indices])
    all_db_entries = list(np.array(all_db_entries)[indices])

    unique_db_entries = []
    unique_ids = set()
    for entry in all_db_entries:
        unique_id = entry.split('-')[-1]
        if unique_id in unique_ids:
            continue
        unique_ids.add(unique_id)
        unique_db_entries.append(entry)

    index: int = 0
    while len(db_entries) < max_entries and index < len(unique_db_entries):
        if unique_db_entries[index] not in db_entries:  # I know, not ideal, but it'll have to do.
            db_entries.append(unique_db_entries[index])
        index += 1
    print(f'Collected {len(db_entries)} database entries.')
    for db_entry in db_entries:
        label: str = os.path.splitext(os.path.basename(db_entry))[0]
        _DATABASE.append(DatabaseEntry(np.load(db_entry), label))


def search(raw_keypoint_file: str, model: Model, k: int) -> Tuple[str, List[str]]:
    """Search through the database with a given file containing keypoints.

    :param raw_keypoint_file: File containing keypoints.
    :param model: SLR model.
    :param k: Top-k results will be returned.
    :return: The label of the keypoint file (ground truth) and the search results."""
    embedding: np.ndarray = model.get_embedding(np.load(raw_keypoint_file))
    with open(raw_keypoint_file.replace('npy', 'json')) as f:
        d: Dict = json.load(f)
        label: str = d['ground_truth']
    search_results: List[str] = get_results(embedding, k)
    return label, search_results


def get_results(embedding: np.ndarray, k: int) -> List[str]:
    distances: List[Tuple[float, int]] = []
    for i, key in enumerate(_DATABASE):
        dist: float = np.linalg.norm(embedding - key.embedding, ord=1)

        distances.append((dist, i))
    ordered = sorted(distances, key=lambda tup: tup[0])  # Sort by ascending distance.
    results: List[str] = [_DATABASE[i].gloss for i in [tup[1] for tup in ordered]][:k]
    return results  # Glosses, distances.

    # distances: List[Tuple[float, int]] = []
    #     for i, key in enumerate(_DATABASE):
    #         dist: float = np.linalg.norm(embedding - key.embedding)
    #
    #         distances.append((eucdist, i))
    #     ordered = sorted(distances, key=lambda tup: tup[0])  # Sort by ascending distance.
    #     results: List[str] = [_DATABASE[i].gloss for i in [tup[1] for tup in ordered]][:k]
    #     return results  # Glosses, distances.

    # distances: List[Tuple[float, int]] = []
    # embedding = embedding.copy()
    # embedding = embedding / np.linalg.norm(embedding)
    # for i, key in enumerate(_DATABASE):
    #     key_embedding = key.embedding.copy()
    #     key_embedding = key_embedding / np.linalg.norm(key_embedding)
    #     cossim = np.dot(embedding, key_embedding)/ (np.linalg.norm(embedding) * np.linalg.norm(key_embedding))
    #
    #     distances.append((cossim, i))
    # ordered = sorted(distances, key=lambda tup: -tup[0])  # Sort by descending similarity.
    # results: List[str] = [_DATABASE[i].gloss for i in [tup[1] for tup in ordered]][:k]
    # return results  # Glosses, distances.


def print_list(array: np.ndarray, indices=(1, 2, 3, 5, 10, 20)) -> str:
    result: str = '['
    for i, el in enumerate(array):
        if i + 1 in indices:  # For the 0th element, prints top-1 accuracy, for the 1st, prints top-2 accuracy...
            result += str(el) + ', '
    result = result[:-2]  # Drop last comma and space.
    result += ']'
    return result


def evaluate(model_name: str, input_directory: str, db_directory: str, model_path: str, k: int, n: int):
    """Convert the videos in `input_directory` to embeddings with the model stored at `model_path` and write them to `output_directory`.

    :param input_directory: The path to the directory containing the dictionary videos.
    :param db_directory: The path to the directory containing the database.
    :param model_path: The path to the model (that generates the embeddings) checkpoint.
    :param k: The top-k accuracy will be computed.
    :param n: The maximum amount of database entries to compare to.
    """
    output = []

    model: Model = Model(model_path)

    populate_db(db_directory, n)

    query_filenames: List[str] = glob.glob(os.path.join(input_directory, '*.npy'))


    # In corpus versus not in corpus.
    incorpus = ['HEBBEN-A-4801', 'PAARD-A-8880', 'STRAAT-A-11560', 'HAAS-A-16146', 'TELEFONEREN-D-11870',
                'HOND-A-5052', 'RUSTEN-B-10250', 'SCHOOL-A-10547', 'ONTHOUDEN-A-8420', 'WAT-A-13657']
    notincorpus = ['BOUWEN-G-1906', 'WAAROM-A-13564', 'MELK-B-7418', 'VALENTIJN-A-16235', 'HERFST-B-4897',
                   'VLIEGTUIG-B-13187', 'KLEPELBEL-A-1166', 'POES-G-9372', 'MOEDER-A-7676', 'VADER-G-8975']

    # Per class accuracy.
    correct = {label.gloss: np.zeros((k,)) for label in _DATABASE[:20]}
    total = {label.gloss: 0 for label in _DATABASE[:20]}
    confusions = {label.gloss: Counter() for label in _DATABASE[:20]}
    for filename in query_filenames:
        label, results = search(filename, model, k)
        for i, result in enumerate(results):
            confusions[label][result] += 1

    print(model_name, 'WAT', confusions['WAT-A-13657'])
    print(model_name, 'VLIEGTUIG', confusions['VLIEGTUIG-B-13187'])
    print(model_name, 'TELEFONEREN', confusions['TELEFONEREN-D-11870'])

if __name__ == '__main__':
    input_directory = '/home/mcdcoste/Documents/research/dagvandewetenschap2023/book/query_keypoints/'
    db_directory_vgt = '/home/mcdcoste/Documents/research/dagvandewetenschap2023/book/db_embeddings_vgt'
    model_vgt = '/home/mcdcoste/Documents/research/dagvandewetenschap2023/book/models/model_vgt.onnx'

    db_directory_autsl = '/home/mcdcoste/Documents/research/dagvandewetenschap2023/book/db_embeddings_autsl'
    model_autsl = '/home/mcdcoste/Documents/research/dagvandewetenschap2023/book/models/model_autsl.onnx'

    import tqdm
    for n in tqdm.tqdm([100]):
        evaluate('VGT', input_directory, db_directory_vgt, model_vgt, 1, n)
        evaluate('AUTSL', input_directory, db_directory_autsl, model_autsl, 1, n)
