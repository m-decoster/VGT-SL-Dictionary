"""Functionality that is common to all datasets, for example, collecting the list of samples."""
import csv
import os
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Sample:
    """A Sample contains a file path, a label and optionally a list of frame indices with which to subsample the input file.
    If the list of frame indices is not provided, the entire file will be loaded."""
    path: str
    label: int
    frame_indices: Optional[List[int]]


def collect_samples(root_path: str, job: str, retrain_on_all: bool) -> List[
    Sample]:
    """Collect a list of samples.

    :param root_path: The dataset root path.
    :param job: "train", "val" or "test".
    :param retrain_on_all: If provided, will load the validation set as part of the training set.
    :return: A list of `Sample`s.
    """
    samples = []
    with open(os.path.join(root_path, 'samples.csv'), 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        assert header == ['Id', 'Label', 'Participant', 'Video', 'Subset']
        for row in reader:
            _id, label, _participant, video, subset = row
            if retrain_on_all and job == 'train':
                if subset == 'train' or subset == 'val':
                    samples.append(Sample(video, int(label), None))
            else:
                if subset == job:
                    samples.append(Sample(video, int(label), None))
    return samples
