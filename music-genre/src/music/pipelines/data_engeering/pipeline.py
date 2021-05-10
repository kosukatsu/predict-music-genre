import os
import sys

sys.path.append("./src/music/pipelines")
from .nodes import (
    load_csv_data,
    split_tempo,
    process_region,
    process_missing,
    normalize_duration,
    normalize_loundness,
    sort_index,
    product_feature,
)
from kedro.pipeline import Pipeline, node

from utils import merge_dictionary

def create_train_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                merge_dictionary,
                ["params:default_preprocess_params","params:train_preprocess_params"],
                "train_preprocess_params"
            ),
            node(split_tempo, "train_raw_data_set", "train_data_splited_tempo"),
            node(
                product_feature,
                [
                    "train_data_splited_tempo",
                    "train_preprocess_params",
                ],
                "train_data_producted_feature"
            ),
            node(
                process_region,
                ["train_data_producted_feature", "train_preprocess_params"],
                "train_data_processed_region",
            ),
            node(
                process_missing,
                [
                    "train_data_processed_region",
                    "train_preprocess_params",
                ],
                "train_data_processed_missing",
            ),

            node(
                normalize_duration,
                [
                    "train_data_processed_missing",
                    "train_preprocess_params",
                ],
                "train_data_normalized_duration",
            ),
            node(
                normalize_loundness,
                [
                    "train_data_normalized_duration",
                    "train_preprocess_params",
                ],
                "train_data_set",
            ),
        ]
    )


def create_test_data_pipeline(**kwargs):
    return Pipeline(
        [
             node(
                merge_dictionary,
                ["params:default_preprocess_params","params:test_preprocess_params"],
                "test_preprocess_params"
            ),           
            node(split_tempo, "test_raw_data_set", "test_data_splited_tempo"),
            node(
                product_feature,
                [
                    "test_data_splited_tempo",
                    "test_preprocess_params",
                ],
                "test_data_producted_feature"
            ),
            
            node(
                process_region,
                ["test_data_producted_feature", "test_preprocess_params"],
                "test_data_processed_region",
            ),
            node(
                process_missing,
                [
                    "test_data_processed_region",
                    "test_preprocess_params",
                ],
                "test_data_filled_missing",
            ),
            node(
                normalize_duration,
                [
                    "test_data_filled_missing",
                    "test_preprocess_params",
                ],
                "test_data_normalized_duration",
            ),
            node(
                normalize_loundness,
                [
                    "test_data_normalized_duration",
                    "test_preprocess_params",
                ],
                "test_data_normalized_loudness",
            ),
            node(sort_index, "test_data_normalized_loudness", "test_data_set"),
        ]
    )
