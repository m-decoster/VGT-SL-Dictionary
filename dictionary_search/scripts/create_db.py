"""This script creates the dictionary vector database by converting videos to embeddings and writing them to disk.."""

import argparse
import glob
import os
from typing import List

import numpy as np
from dictionary_search_model import Model, load_video, preprocess_video
from joblib import Parallel, delayed


def _process(video_file: str, output_directory: str, model_path: str, extension: str) -> bool:
    """Convert a single video file.

    :param video_file: The path to the video file.
    :param output_directory: The path to the output directory to which embeddings will be written.
    :param model_path: The path to the model (that generates the embeddings) checkpoint.
    :param extension: The video file extension.
    :return: True if it was extracted, False if the file already existed or the video could not be opened."""
    output_filename: str = os.path.basename(video_file).replace(extension, '.npy')
    output_path: str = os.path.join(output_directory, output_filename)

    if os.path.exists(output_path):
        return False

    # We load the model here because the ONNX InferenceSession is not pickleable, so it does not work with joblib.
    # It still appears to be quite fast.
    model: Model = Model(model_path)

    try:
        frames: np.ndarray = load_video(video_file)
        features: np.ndarray = preprocess_video(frames)
        embedding: np.ndarray = model.get_embedding(features)
    except FileNotFoundError:
        return False

    np.save(output_path, embedding)

    return True


def main(input_directory: str, output_directory: str, model_path: str, extension: str, jobs: int):
    """Convert the videos in `input_directory` to embeddings with the model stored at `model_path` and write them to `output_directory`.

    :param input_directory: The path to the directory containing the dictionary videos.
    :param output_directory: The path to the output directory to which embeddings will be written.
    :param model_path: The path to the model (that generates the embeddings) checkpoint.
    :param extension: The video file extension.
    :param jobs: Parallel processes.
    """
    if extension[0] != '.':
        extension = f'.{extension}'

    video_files: List[str] = glob.glob(os.path.join(input_directory, f'*{extension}'))

    os.makedirs(output_directory, exist_ok=True)

    _results = Parallel(n_jobs=jobs)(
        delayed(_process)(video_file, output_directory, model_path, extension) for video_file in video_files)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('input_directory', type=str, help='The path to the directory containing the dictionary videos.')
    parser.add_argument('output_directory', type=str,
                        help='The path to the output directory to which embeddings will be written.')
    parser.add_argument('model_path', type=str,
                        help='The path to the model (that generates the embeddings) checkpoint.')
    parser.add_argument('-e', '--extension', type=str, help='The video file name extension.', default='mp4')
    parser.add_argument('-j', '--jobs', type=int, help='Number of parallel processes.', default=-1)

    args = parser.parse_args()

    main(args.input_directory, args.output_directory, args.model_path, args.extension, args.jobs)
