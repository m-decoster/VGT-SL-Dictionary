FROM ubuntu:mantic

RUN apt-get update
RUN apt-get install -y python3 python3-pip python3-venv

WORKDIR /dictionary_search

COPY . .

WORKDIR /

RUN python3 -m venv /pipenv
RUN /pipenv/bin/pip3 install -e dictionary_search

EXPOSE 5000

WORKDIR /dictionary_search/demo_webapp

CMD /pipenv/bin/python3 app.py "${MODEL_PATH}" "${OUTPUT_PATH}" "${DB_PATH}"
