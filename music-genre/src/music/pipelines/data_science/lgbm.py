import os, sys

sys.path.append("./src/music/pipelines")

import logging

import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
import optuna
import optuna.integration.lightgbm as opt_lgb


def cross_validation_model(model_params, train_x, train_y, k, seed):
    logger = logging.getLogger(__name__)
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)

    model_params["seed"] = seed

    accuracys = []

    for train_idx, val_idx in kf.split(train_x):
        train_data = lgb.Dataset(train_x.iloc[train_idx], label=train_y[train_idx])
        valid_data = lgb.Dataset(train_x.iloc[val_idx], label=train_y[val_idx])

        lgbm = lgb.train(model_params, train_data, valid_sets=valid_data,)

        preds = lgbm.predict(train_x.iloc[val_idx])
        accuracy = accuracy_score(preds.argmax(axis=1), train_y[val_idx])
        accuracys.append(accuracy)

    accuracys = np.array(accuracys)
    accuracy = accuracys.mean()
    logger.info("accuracy:%f", accuracy)
    return accuracy


def tuning_objective(trial, model_params, train_x, train_y, k, seed, tuning_params):
    for params in tuning_params["use_params"]:
        if "log" in tuning_params[params]:
            use_log = tuning_params[params]["log"]
        else:
            use_log = False
        type_name = tuning_params[params]["type"]
        lower = tuning_params[params]["lower"]
        upper = tuning_params[params]["upper"]
        if type_name == "int":
            model_params[params] = trial.suggest_int(params, lower, upper)
        elif type_name == "float":
            model_params[params] = trial.suggest_float(
                params, lower, upper, log=use_log
            )

    return cross_validation_model(model_params, train_x, train_y, k, seed)


def hyper_parameter_tuning(model_params, train_x, train_y, k, seed, tuning_params):
    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: tuning_objective(
            trial, model_params, train_x, train_y, k, seed, tuning_params
        ),
        tuning_params["n_trials"],
    )

    trial = study.best_trial

    print("Best accuracy:{}", format(trial.value))
    print("Best params:")
    for key, value in trial.params.items():
        print("\t{}:\t{}".format(key, value))

    return trial


def train(model_params, train_x, train_y, seed, train_rate, save_model_path):
    logger = logging.getLogger(__name__)

    model_params["seed"] = seed

    data_len = len(train_x)
    train_idx = np.ones(data_len, dtype=bool)
    train_idx[int(data_len * train_rate) :] = False
    np.random.seed(seed)
    np.random.shuffle(train_idx)
    val_idx = train_idx == False

    train_data = lgb.Dataset(train_x[train_idx], label=train_y[train_idx])
    valid_data = lgb.Dataset(train_x[val_idx], label=train_y[val_idx])

    lgbm = lgb.train(model_params, train_data, valid_sets=valid_data)

    preds = lgbm.predict(train_x.iloc[val_idx])
    accuracy = accuracy_score(preds.argmax(axis=1), train_y[val_idx])
    logger.info("accuracy:%f", accuracy)

    return lgbm


def predict(test_data, lgbm):
    preds = lgbm.predict(test_data)
    return pd.DataFrame([np.arange(4046, 4046 * 2), preds.argmax(axis=1)]).T


def select_feature(lgbm, params, train_x, train_y):
    selector = SelectFromModel(lgb.LGBMClassifier(**params), threshold="median")
    selector.fit(train_x, train_y)
    print(train_x.columns)
    print(selector.get_support())
    print(train_x.columns[selector.get_support()])
    print(selector.estimator_.feature_importances_)
    return dict(zip(train_x.columns, selector.estimator_.feature_importances_))
