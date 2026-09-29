"""Tests for the data processing functions."""

import pandas as pd

from src.data_processing import prepare_model_data


def test_prepare_model_data_split():
    """
    Test that the dataset is divided into the expected training and test sizes.
    """
    # Create a small representative dataset with 10 observations
    data = pd.DataFrame({
        "frequency": range(10),
        "angle_of_attack": range(10),
        "chord_length": range(10),
        "free_stream_velocity": range(10),
        "displacement_thickness": range(10),
        "sound_pressure_level": range(10),
    })

    X_train, X_test, y_train, y_test = prepare_model_data(
        data,
        test_size=0.20,
        random_state=42
    )

    # An 80/20 split of 10 observations should produce 8 training and 2 test rows
    assert len(X_train) == 8
    assert len(X_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2
