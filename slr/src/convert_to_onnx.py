"""Convert a model checkpoint to ONNX."""
import argparse

import torch
from torch import nn

from models.module import Module


def convert(ptl_model_checkpoint_path: str, output_path: str):
    """Load a PyTorch Lightning model checkpoint and convert it to an ONNX model file, which
    will be saved at the specified path.

    :param ptl_model_checkpoint_path: Path to the PyTorch Lightning checkpoint on disk.
    :param output_path: Path to the output file."""
    module: Module = Module.load_from_checkpoint(ptl_model_checkpoint_path)
    module = module.eval()
    model: nn.Module = module.model

    # The embeddings we want are the sequence embeddings.
    model.head = nn.Identity()

    # First axis: time (dynamic).
    # Second axis: batch (fixed at 1 for inference).
    # Third axis: number of keypoints (75).
    # Last axis: number of coordinates per keypoint (3).
    dummy_input: torch.Tensor = torch.zeros((50, 1, 75, 3))

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        opset_version=11,
        dynamic_axes={"input": {0: "input"}}
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('pytorch_checkpoint', type=str, help='The path to PyTorch Lightning checkpoint.')
    parser.add_argument('output', type=str, help='The path to the output ONNX file.')

    args = parser.parse_args()

    convert(args.pytorch_checkpoint, args.output)
