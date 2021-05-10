import pickle

import pandas as pd


def split_data(data):
    y = data["genre"].to_numpy()
    x = data.drop("genre", axis=1).copy()
    return x, y

def drop_feature(data,feature_dropping):
    return data.drop(feature_dropping,axis=1).copy()

def merge_dictionary(dict1,dict2):
    dict1.update(dict2)
    return dict1

def merge_pseudo_data(data1,data2,use_pseudo):
    if use_pseudo:
        df=pd.concat([data1,data2])
        df["genre"]=df["genre"].astype(int)
        return df
    else:
        return data1