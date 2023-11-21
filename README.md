# Flemish Sign Language Dictionary Search Functionality Prototype

At
the [Dag Van De Wetenschap 2023](https://www.dagvandewetenschap.be/activiteiten/universiteit-gent-kunnen-computers-gebarentaal-begrijpen-op-locatie), [IDLab-AIRO](https://airo.ugent.be/)
demonstrated a prototype of a search tool for the [VGT-NL](https://woordenboek.vlaamsegebarentaal.be/) dictionary.

This prototype is based on [this paper](https://users.ugent.be/~mcdcoste/assets/2023095125.pdf), with an updated sign
language recognition model that has better performance. A further notable improvement is obtained by
running MediaPipe in the browser instead of the backend, leading to near-instant search results (down to tens
of milliseconds from several seconds).

In this repository, you will find:

- Code to train the sign language recognition model (`slr/`).
- A link to the model checkpoint (see below).
- Code to create a vector database from the dictionary (`dictionary_search/scripts/create_db.py`).
- Code of the prototype, including frontend and backend. This includes a data collection
  pipeline (`dictionary_search/demo_webapp/`).
- Code to evaluate the search system with a set of query videos (`dictionary_search/scripts/evaluate.py`).

## Branches

This demo has been repurposed for several events. Some of these events were in Dutch, some in English.

- `DagVanDeWetenschap2023` branch [Dutch]: For the Dag Van De Wetenschap on November 26th 2023.
- `SignON_EASIER_EC` branch [English]: For the SignON - EASIER workshop at the European Commission on November 29th 2023.

## Installation

There are two main parts to this repository: the demo application and the sign language recognition model.
We provide these as separate environments, to reduce the size of the dependencies of the demo application
(and avoid pulling the entirety of PyTorch).

The installation and usage of both parts are similar and described below.

The docker files use virtual environments located in the container under `/pipenv`. To run Python commands,
use `/pipenv/bin/python3` as the interpreter.

### Sign language recognition model

The sign language recognition in `slr/` model is built with PyTorch. You can install its dependencies
with docker or in a virtual environment.

The model was built using the [SignON](https://signon-project.eu) data sets, which are not yet publicly available.
They will become available at the end of 2023.

#### Using docker

See `slr/Dockerfile` for details on the Docker image.
When running the container, make sure to:

1. Set the `WANDB_API_KEY` environment variable to enable logging to Weights and Biases
2. Mount the required volume(s) to the Docker container (for logging and data loading)
3. Run the desired script, `train.py`, `test.py` or `convert_to_onnx.py` with the required command line arguments. See
   `slr/README.md` for more details

#### Locally

Create a virtual environment (Python 3) and install the dependencies using `pip install -e slr`.

### Demo application

The demo application in `dictionary_search` has a Flask backend and a very basic HTML frontend.
The frontend of the web application is built with vanilla HTML, CSS and JavaScript.
No JavaScript build tools are required.

The demo application can also be installed with docker or in a virtual environment.

To create the database, you should first download all dictionary videos to a single directory.
The VGT dictionary videos were gathered as part of the [SignON](https://signon-project.eu) research project.
They can be downloaded from [INT](https://taalmaterialen.ivdnt.org/download/woordenboek-vgt/).

#### Using docker

See `dictionary_search/Dockerfile` for details on the Docker image.
When running the container, the application will automatically be started.
When running the container, make sure to:

1. Set the `DB_PATH` environment variable to the location of your database, created with `scripts/create_db.py`
   (see below)
2. Set the `OUTPUT_PATH` environment variable to the location to which you wish to write output files
3. Set the `MODEL_PATH` environment variable to the location of the ONNX model file.
4. Mount the required volume(s) to the Docker container (for storing and data loading)
5. Use the `-P` flag to forward port `5000`
6. Use the `--network host` command line argument to allow connecting to `localhost:5000` from the host. This is
   the easiest way to access the webcam from the server

For example:

```commandline
docker build -t dictionary_search .
docker run --network host -P --name demo -v /host/path/to/volume:/db -e MODEL_PATH=/db/model.onnx -e OUTPUT_PATH=/db/out -e DB_PATH=/db/db_embeddings dictionary_search
```

#### Locally

Create a virtual environment (Python 3) and install the dependencies using `pip install -e dictionary_search`.
See `Dockerfile` for an invocation of the application.

#### Creating the database

To create the database, run `scripts/create_db.py` in the virtual environment or docker container.
This script should be fairly self-explanatory: it uses the provided ONNX model to convert the videos in one folder
to embeddings in another.

## License

The code in this repository is licensed under the MIT license. See `LICENSE`.

We use [MediaPipe](https://developers.google.com/mediapipe), which is licensed under the Apache 2.0 license.
See `dictionary_search/demo_webapp/frontend/mediapipe/LICENSE`.

## Citation

If you find this code useful, please consider citing our work:

```
@inproceedings{de2023querying,
  title={Querying a sign language dictionary with videos using dense vector search},
  author={De Coster, Mathieu and Dambre, Joni},
  booktitle={2023 IEEE International Conference on Acoustics, Speech, and Signal Processing Workshops (ICASSPW)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```

The underlying SLR model is the PoseFormer created by [Maxim Bonnaerens](https://maxim.bonnaerens.com/),
[Joni Dambre](https://airo.ugent.be/members/joni/) and [Mathieu De Coster](https://users.ugent.be/~mcdcoste/index.html)
for our participation in [Google's recent Kaggle competition](https://www.kaggle.com/competitions/asl-signs)
where we finished 16th out of 1165 teams.
The checkpoint is available for download from [here](https://cloud.ilabt.imec.be/index.php/s/RmySLwxaAGWp5ye).
