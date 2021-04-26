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
)
from kedro.pipeline import Pipeline, node


def create_train_data_pipeline(**kwargs):
    return Pipeline(
        [
            # node(load_csv_data, "params:train_raw_data_path", "loaded_train_data"),
            node(split_tempo, "train_raw_data_set", "train_data_splited_tempo"),
            node(
                process_region,
                ["train_data_splited_tempo", "params:train_preprocess_params"],
                "train_data_processed_region",
            ),
            node(
                process_missing,
                [
                    "train_data_processed_region",
                    "params:train_preprocess_params.process_missing",
                ],
                "train_data_processed_missing",
            ),
            node(
                normalize_duration,
                [
                    "train_data_processed_missing",
                    "params:train_preprocess_params.max_duration_ms",
                ],
                "train_data_normalized_duration",
            ),
            node(
                normalize_loundness,
                [
                    "train_data_normalized_duration",
                    "params:train_preprocess_params.max_loundness_dB",
                ],
                "train_data_set",
            ),
        ]
    )


def create_test_data_pipeline(**kwargs):
    return Pipeline(
        [
            # node(load_csv_data, "params:test_raw_data_path", "loaded_test_data"),
            node(split_tempo, "test_raw_data_set", "test_data_splited_tempo"),
            node(
                process_region,
                ["test_data_splited_tempo", "params:test_preprocess_params"],
                "test_data_processed_region",
            ),
            node(
                process_missing,
                [
                    "test_data_processed_region",
                    "params:test_preprocess_params.process_missing",
                ],
                "test_data_filled_missing",
            ),
            node(
                normalize_duration,
                [
                    "test_data_filled_missing",
                    "params:test_preprocess_params.max_duration_ms",
                ],
                "test_data_normalized_duration",
            ),
            node(
                normalize_loundness,
                [
                    "test_data_normalized_duration",
                    "params:test_preprocess_params.max_loundness_dB",
                ],
                "test_data_normalized_loudness",
            ),
            node(sort_index, "test_data_normalized_loudness", "test_data_set"),
        ]
    )
