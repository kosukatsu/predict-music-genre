from kedro.pipeline import Pipeline, node

from .nodes import load_data, split_tempo, drop_region, fill_missing, split_data, normalize_duration, normalize_loundness


def create_train_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                load_data,
                "params:train_data_path",
                "loaded_train_data"
            ),
            node(
                split_tempo,
                "loaded_train_data",
                "train_data_splited_tempo"
            ),
            node(
                drop_region,
                "train_data_splited_tempo",
                "train_data_dropped_region"
            ),
            node(
                fill_missing,
                "train_data_dropped_region",
                "train_data_filled_missing"
            ),
            node(
                normalize_duration,
                ["train_data_filled_missing", "params:max_duration_ms"],
                "train_data_normalized_duration"
            ),
            node(
                normalize_loundness,
                ["train_data_normalized_duration", "params:max_loundness_dB"],
                "train_data_normalized_loundness"
            ),
            node(
                split_data,
                ["train_data_normalized_loundness"],
                ["train_x", "train_y"],
            )
        ]
    )


def create_test_data_pipeline(**kwargs):
    return Pipeline(
        [
            node(
                load_data,
                "params:test_data_path",
                "loaded_test_data"
            ),
            node(
                split_tempo,
                "loaded_test_data",
                "test_data_splited_tempo"
            ),
            node(
                drop_region,
                "test_data_splited_tempo",
                "test_data_dropped_region"
            ),
            node(
                fill_missing,
                "test_data_dropped_region",
                "test_data_filled_missing"
            ),
            node(
                normalize_duration,
                ["train_data_filled_missing", "params:max_duration_ms"],
                "train_data_normalized_duration"
            ),
            node(
                normalize_loundness,
                ["train_data_normalized_duration", "params:max_loundness_dB"],
                "train_data_normalized_loundness"
            )
        ]
    )
