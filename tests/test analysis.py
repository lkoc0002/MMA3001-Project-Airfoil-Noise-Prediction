"""Tests for the engineering sensitivity analysis functions."""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from src.analysis import sensitivity_prediction


def test_sensitivity_prediction_output():
    """
    Test the output of the sensitivity prediction function.
    """
    # Create a small dataset containing two engineering input variables
    data = pd.DataFrame({
        "frequency": [200, 400, 600, 800, 1000],
        "free_stream_velocity": [30, 40, 50, 60, 70],
    })

    # Create a simple target with a predictable relationship to the inputs
    y = (
        0.01 * data["frequency"]
        + 0.1 * data["free_stream_velocity"]
    )

    model = LinearRegression()
    model.fit(data, y)

    variable_values, predictions = sensitivity_prediction(
        model,
        data,
        "frequency",
        points=20
    )

    # The function should return the requested number of sensitivity points
    assert len(variable_values) == 20
    assert len(predictions) == 20

    # The sensitivity range should match the observed frequency range
    assert np.isclose(variable_values[0], data["frequency"].min())
    assert np.isclose(variable_values[-1], data["frequency"].max())

    # Frequency has a positive coefficient, so predictions should increase
    assert predictions[-1] > predictions[0]
