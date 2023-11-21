FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

WORKDIR /slr

COPY . .

WORKDIR /

RUN python3 -m venv /pipenv
RUN /pipenv/bin/pip3 install -e dictionary_search

RUN /pipenv/bin/python3 -m pip install -e slr
