from flask import Flask
from flask import request
import numpy as np
import pandas as pd
from chatbot import get_response
from movie_rec_routine import movie_rec

similarity = pd.read_parquet('similarity', engine='pyarrow')
similarity = np.array(similarity)

app = Flask(__name__)

@app.route('/getResponse')
def getResponse():
    text = request.args.get('text')
    if text[:6].lower() == 'movie:':
        message = {'answer': movie_rec(text[6:], similarity)}
    else:
        response = get_response(text)
        message = {'answer': response}
    return message

if __name__ == '__main__':
    app.run()