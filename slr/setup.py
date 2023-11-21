import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="slr",
    version="0.0.1",
    description="A sign language recognition model.",
    author="Mathieu De Coster",
    author_email="mathieu.decoster@ugent.be",
    install_requires=["numpy==1.24.1",
                      "mediapipe==0.9.3.0",
                      "onnx==1.14.1",
                      "opencv-contrib-python==4.7.0.72",
                      "torch==1.12.1",
                      "pytorch_lightning==1.7.7",
                      "wandb==0.15.11",
                      "scikit-learn==1.1.2",
                      "torchmetrics==0.10.1"],
    packages=setuptools.find_packages(),
)
