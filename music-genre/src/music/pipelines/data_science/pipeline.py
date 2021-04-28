import os, sys

sys.path.append("./src/music/pipelines")

from kedro.pipeline import Pipeline, node

from .lgbm import (
    cross_validation_model,
    hyper_parameter_tuning,
    train,
    predict,
    select_feature,
)
from utils import split_data


def create_cross_validation_pipeline(**kwargs):
    return Pipeline(
        [
            node(split_data, "train_data_set", ["train_x", "train_y"],),
            node(
                cross_validation_model,
                {
                    "model_params": "params:lgbm_params",
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "k": "params:cross_validation_k",
                    "seed": "params:seed",
                },
                "accuracy",
            ),
        ]
    )


def create_hy_para_tuning_pipeline(**kwargs):
    return Pipeline(
        [
            node(split_data, "train_data_set", ["train_x", "train_y"],),
            node(
                hyper_parameter_tuning,
                {
                    "model_params": "params:lgbm_params",
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "k": "params:cross_validation_k",
                    "seed": "params:seed",
                    "tuning_params": "params:lgbm_hyper_parameter_tuning",
                },
                "lgbm_model_hypara_tuning",
            ),
        ]
    )


def create_real_train_pipeline(**kwargs):
    return Pipeline(
        [
            node(split_data, "train_data_set", ["train_x", "train_y"],),
            node(
                train,
                {
                    "model_params": "params:real_lgbm_params",
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "seed": "params:seed",
                    "train_rate": "params:train_rate",
                    "save_model_path": "params:model_path",
                },
                "lgbm_model",
            ),
        ],
    )


def create_eval_pipeline(**kwargs):
    return Pipeline([node(predict, ["test_data_set", "lgbm_model"], "lgbm_output"),])


def create_select_feature_pipeline(**kwargs):
    return Pipeline(
        [
            node(split_data, "train_data_set", ["train_x", "train_y"]),
            node(
                select_feature,
                ["lgbm_model", "params:real_lgbm_params", "train_x", "train_y"],
                None,
            ),
        ]
    )

