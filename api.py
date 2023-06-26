from flask import Flask
from flask import request, jsonify, Response
import numpy as np
import pandas as pd
from chatbot import get_response
from movie_rec_routine import movie_rec
from flask_cors import CORS

similarity = pd.read_parquet('similarity', engine='pyarrow')
similarity = np.array(similarity)

app = Flask(__name__)

CORS(app)


@app.route('/data', methods=['GET'])
def data():
    text = request.args.get('text')
    if text[:6].lower() == 'movie:':
        message = {'answer': movie_rec(text[6:], similarity)}
    else:
        response = get_response(text)
        message = {'answer': response}

    return message


if __name__ == '__main__':
    app.run(debug=True)
