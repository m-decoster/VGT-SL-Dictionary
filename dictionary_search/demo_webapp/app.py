import argparse
import json
import os
import uuid
from typing import List, Dict, Tuple

import numpy as np
from dictionary_search_model import Model
from flask import Flask, request, Response
from flask.json import jsonify
from flask_cors import CORS
from werkzeug.utils import redirect

import search_engine

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str,
                    help='The path to the model (that generates the embeddings) checkpoint.')
parser.add_argument('data_store', type=str, help='Where to save recordings.')
parser.add_argument('db_path', type=str, help='The path to the database directory.')
parser.add_argument('-m', '--max_entries', type=int, help='Maximum database entries to consider.', default=100)
args = parser.parse_args()

model: Model = Model(args.model_path)
database: search_engine.SearchEngine = search_engine.SearchEngine(args.db_path, args.max_entries)

app = Flask(__name__, static_url_path='', static_folder='frontend')
CORS(app)


@app.route('/')
def index():
    return redirect("index.html", code=302)


@app.route("/search", methods=['post'])
def search() -> Response:
    keypoints = np.fromstring(request.data, dtype=float, sep=',')
    features = keypoints.reshape((-1, 543, 3))

    embedding: np.ndarray = model.get_embedding(features)
    search_results: List[Tuple[str, float]] = database.get_results(embedding)

    unique_filename: uuid.UUID = uuid.uuid4()
    np.save(os.path.join(args.data_store, str(unique_filename)), features)

    with open(os.path.join(args.data_store, str(unique_filename) + '.json'), 'w') as metadata_file:
        json.dump({'ground_truth': request.args.get('gtgloss'),
                   'keypoints_file': str(unique_filename),
                   'predictions': search_results}, metadata_file)

    # Return search results.
    response: Dict[str, List[str]] = {
        "results": [result[0] for result in search_results[:9]]
        # Only return 9 glosses to the UI, which can only display 9 anyway.
    }

    response: Response = jsonify(response)

    return response


if __name__ == '__main__':
    app.run(debug=False, threaded=False, host='localhost', port=5000)
