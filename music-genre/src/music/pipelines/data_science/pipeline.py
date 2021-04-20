from kedro.pipeline import Pipeline, node

from .lgbm import cross_validation_model


def create_cross_validation_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                cross_validation_model,
                {"model_params": "params:lgbm_params",
                 "train_x": "train_x",
                 "train_y": "train_y",
                 "k": "params:cross_validation_k",
                 "seed": "params:seed",
                 "train_params": "params:lgbm_train_params"
                 },
                "accuracy"
            )
        ]
    )
