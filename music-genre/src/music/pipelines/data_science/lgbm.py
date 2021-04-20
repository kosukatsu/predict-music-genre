import logging

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import numpy as np


def cross_validation_model(model_params, train_x, train_y, k, seed, train_params):
    logger = logging.getLogger(__name__)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    accuracys = []

    for train_idx, val_idx in kf.split(train_x):
        train_data = lgb.Dataset(train_x.iloc[train_idx], label=train_y[train_idx])
        valid_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y[val_idx])

        lgbm = lgb.train(
            model_params,
            train_data,
            valid_sets=valid_data,
            **train_params
        )

        preds = lgbm.predict(train_x.iloc[val_idx])
        accuracy = accuracy_score(preds.argmax(axis=1), train_y[val_idx])
        accuracys.append(accuracy)

    accuracys = np.array(accuracys)
    accuracy = accuracys.mean()
    logger.info("accuracy:%f", accuracy)
    return accuracy
