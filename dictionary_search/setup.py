import pathlib

import setuptools

root_folder = pathlib.Path(__file__).parents[1]
setuptools.setup(
    name="dictionary_search",
    version="0.0.1",
    description="A sign language dictionary search system powered by a sign language recognition model.",
    author="Mathieu De Coster",
    author_email="mathieu.decoster@ugent.be",
    install_requires=["numpy==1.24.2",
                      "onnxruntime==1.16.1",
                      "flask==3.0.0",
                      "flask-cors==4.0.0",
                      "joblib==1.3.2",
                      "tqdm==4.64.1",
                      "opencv-contrib-python==4.7.0.72",
                      "mediapipe==0.9.3.0"],
    packages=setuptools.find_packages(),
)
