import pickle

import pandas as pd


def split_data(data):
    y = data["genre"].to_numpy()
    x = data.drop("genre", axis=1).copy()
    return x, y

