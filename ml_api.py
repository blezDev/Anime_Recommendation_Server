import json
import pickle
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import random
import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact
from fastapi import FastAPI
from pydantic import BaseModel
import requests
app = FastAPI()


# loading the saved model
def load_model():
    return pd.read_pickle('anime_model.pkl')


class model_input(BaseModel):
    anime: str


# Converts the pivot table to a sparse matrix


def get_recommendations(user_input):
    anime_model = load_model()
    anime_matrix = csr_matrix(anime_model.values)

    # Fit nearest neighbors model
    anime_nn = NearestNeighbors(metric='cosine', algorithm='brute')
    anime_nn.fit(anime_matrix)

    try:
        query_index = anime_model[anime_model.index == user_input].iloc[0]
        # print(query_index)
        distances, indices = anime_nn.kneighbors(query_index.values.reshape(1, -1), n_neighbors=11)
        # print(indices)
        # Output the recommended shows in a user readable format
        anime_list = []
        for i in range(0, len(distances.flatten())):
            anime_list.append(anime_model.index[indices.flatten()[i]])
        return anime_list
    except Exception as e:
        return "Anime name not found."
        # print(e)


def capitalize_first_letter(s):
    words = s.split(' ')
    capitalized_words = [word.capitalize() for word in words]
    return ' '.join(capitalized_words)


@app.get("/")
def hello():
    return "Welcome to anime recommendation api"


@app.post("/predict")
def getRecommendation(input: model_input):
    input_data = input.json()
    input_dictionary = json.loads(input_data)
    anime = input_dictionary["anime"]
    anime_array = get_recommendations(capitalize_first_letter(anime))
    object = {"query" : anime_array}
    return object



