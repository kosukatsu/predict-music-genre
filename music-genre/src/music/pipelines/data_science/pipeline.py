from kedro.pipeline import Pipeline, node

from .lgbm import cross_validation_model, hyper_parameter_tuning


def create_cross_validation_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                cross_validation_model,
                {
                    "model_params": "params:lgbm_params",
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "k": "params:cross_validation_k",
                    "seed": "params:seed",
                },
                "accuracy"
            )
        ]
    )


def create_hy_para_tuning_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                hyper_parameter_tuning,
                {
                    "model_params": "params:lgbm_params",
                    "train_x": "train_x",
                    "train_y": "train_y",
                    "k": "params:cross_validation_k",
                    "seed": "params:seed",
                    "tuning_params": "params:lgbm_hyper_parameter_tuning"
                },
                []
            )
        ]
    )
