import pandas as pd
import numpy as np


def load_data(data_file):
    data = pd.read_csv(data_file, index_col=0)
    return data


def split_tempo(data):
    tempo = data["tempo"].str.split("-", expand=True).astype(int)
    tempo.columns = ["lower_tempo", "upper_tempo"]
    data = pd.merge(data, tempo, left_index=True, right_index=True)
    data = data.drop("tempo", axis=1)
    return data


def drop_region(data):
    data = data.drop("region", axis=1)
    return data


def fill_missing(data):
    data = data.fillna(data.mean())
    return data


def normalize_duration(data, peak_value):
    duration = data["duration_ms"].to_numpy()
    duration[duration > peak_value] = peak_value
    duration = duration.astype(float) / peak_value
    data["duration"] = duration
    data = data.drop("duration_ms", axis=1)
    return data


def normalize_loundness(data, peak_value):
    loundness = data["loudness"].to_numpy()
    loundness = -loundness
    loundness[loundness > peak_value] = peak_value
    loundness[loundness < 0] = 0
    loundness = loundness / peak_value
    data["loudness"] = loundness
    return data


def split_data(data):
    y = data["genre"].to_numpy()
    x = data.drop("genre", axis=1)
    return x, y
