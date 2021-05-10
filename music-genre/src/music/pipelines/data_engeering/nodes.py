import pickle

import pandas as pd
import numpy as np


def load_csv_data(data_file):
    data = pd.read_csv(data_file, index_col=0)
    return data


def split_tempo(data):
    tempo = data["tempo"].str.split("-", expand=True).astype(int)
    tempo.columns = ["lower_tempo", "upper_tempo"]
    data = pd.merge(data, tempo, left_index=True, right_index=True)
    data["tempo_range"] = tempo["upper_tempo"] - tempo["lower_tempo"]
    data = data.drop("tempo", axis=1)
    return data


def process_region(data, preprocess_params):
    process_name = preprocess_params["process_region"]
    if process_name == "drop":
        data = data.drop("region", axis=1)
    elif process_name == "replace-aggregate":
        # replace
        replaces = preprocess_params["regions_as_unknown"]
        data = data.replace(replaces, "unknown")
        if "regions_white_list" in preprocess_params:
            regions_black_list = np.setdiff1d(
                data["region"].unique(), preprocess_params["regions_white_list"]
            )
            data = data.replace(regions_black_list, "unknown")

        # aggregate
        region_list = data["region"].unique()
        new_datas = []
        for region in region_list:
            data_per_region = data[data["region"] == region]
            if region == "unknown":
                agg_data = data.drop("region", axis=1).agg(["std", "mean"])
            else:
                agg_data = data_per_region.drop("region", axis=1).agg(["std", "mean"])
            new_data = (
                data_per_region.drop("region", axis=1) - agg_data.loc["mean"]
            ) / (agg_data.loc["std"] + 1e-8)
            if "genre" in new_data.columns:
                new_data = new_data.drop("genre", axis=1)
            new_data.columns = "z_score_per_reigon_" + new_data.columns
            new_data = pd.merge(
                data_per_region,
                new_data,
                how="outer",
                left_index=True,
                right_index=True,
            )
            new_datas.append(new_data)

        data = pd.concat(new_datas)
        data["region"] = data["region"].astype("category")
    return data


def process_missing(data,params):
    process_name=params["process_missing"]
    if process_name == "drop":
        data.dropna(how="any")
    elif process_name == "mean":
        data = data.fillna(data.mean())
    return data


def normalize_duration(data, params):
    peak_value=params["max_duration_ms"]
    duration = data["duration_ms"].to_numpy()
    duration[duration > peak_value] = peak_value
    duration = duration.astype(float) / peak_value
    data["duration"] = duration
    data = data.drop("duration_ms", axis=1)
    return data


def normalize_loundness(data, params):
    peak_value=params["max_loundness_dB"]
    loundness = data["loudness"].to_numpy()
    loundness = -loundness
    loundness[loundness > peak_value] = peak_value
    loundness[loundness < 0] = 0
    loundness = loundness / peak_value
    data["loudness"] = loundness
    return data


def sort_index(data):
    return data.sort_index()


def product_feature(data,params):
    features=params["feature_product"]
    features_num=len(features)
    for i in range(features_num):
        for j in np.arange(start=i+1,stop=features_num):
            column_name="sqrt_{}_x_{}".format(features[i],features[j])
            data[column_name]=data[features[i]]*data[features[j]]
            data[column_name].apply(np.sqrt)
    return data