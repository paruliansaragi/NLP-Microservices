from flask import Flask, request
from flask_restful import reqparse, abort, Api, Resource
import pickle
import numpy as np
from keras.models import load_model
#from model import PersonalityModel
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

app = Flask(__name__)
api = Api(app)

#model = NLPModel()

# To do: convert words to vectors, create NLP model 

# argument parsing
parser = reqparse.RequestParser()
parser.add_argument('query')
# The parser will look through the parameters that a user sends to your API. 
# The parameters will be in a Python dictionary or JSON object. 
# For this example, we will be specifically looking for a key called query. 
# The query will be a phrase that a user will want our model to make a prediction 
# on whether the phrase is positive or negative.


# Resources are the main building blocks for Flask RESTful APIs. Each class can have methods 
# that correspond to HTTP methods such as: GET, PUT, POST, and DELETE. GET will be the primary 
# method because our objective is to serve predictions. In the get method below, we provide 
# directions on how to handle the user’s query and how to package the JSON object that will be 
# returned to the user.


# Load your model with its weights
# from keras.models import load_model
# new_model = load_model(filepath)
# model.load_weights('my_model_weights.h5')
# Preprocess your data

# Perform the actual prediction
# Handle the prediction response data

class PredictPersonality(Resource):
    def get(self):
        # use parser and find the user's query
        args = parser.parse_args()
        user_query = args['query']
        
        model = load_model('personality-detect.h5')
        model.load_weights('pd-weights.h5') 
        
        form_data = [user_query]
        list_tokenized = tokenizer.texts_to_sequences(form_data)
        X_ex = pad_sequences(list_tokenized, maxlen=100)
        preds = model.predict([X_ex])
        def f(x):
            return {
                9: 'INFP',
                8:'INFJ',
                11:'INTP',
                10:'INTJ',
                3:'ENTP',
                1:'ENFP',
                15:'ISTP',
                13:'ISFP',
                2:'ENTJ',
                14:'ISTJ',
                0:'ENFJ',
                12:'ISFJ',
                7:'ESTP',
                5:'ESFP',
                4:'ESFJ',
                6:'ESTJ'
            }[x]
        answer = f(np.argmax(preds))
        tf.keras.backend.clear_session()#essential
        # vectorize the user's query and make a prediction
        '''
        '''
        return {'prediction': answer}

#curl -X GET http://127.0.0.1:5000/ -d query='that movie was boring'
# Setup the Api resource routing here
# Route the URL to the resource
api.add_resource(PredictPersonality, '/')

#run a virtual env
#
#cd../ .\venv\Scripts\activate
#curl a response

if __name__ == '__main__':
    app.run(debug=True)