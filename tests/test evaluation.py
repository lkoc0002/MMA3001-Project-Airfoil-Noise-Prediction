"""Tests for the regression evaluation functions."""

import numpy as np

from src.evaluation import evaluate_regression


def test_evaluate_regression_perfect_prediction():
    """
    Test regression metrics for a perfect set of predictions.
    """
    # Create measured and predicted values that are identical
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0])

    metrics = evaluate_regression(y_true, y_pred)

    # Perfect predictions should have zero error and an R-squared value of one
    assert np.isclose(metrics["MAE"], 0.0)
    assert np.isclose(metrics["RMSE"], 0.0)
    assert np.isclose(metrics["R2"], 1.0)


def test_evaluate_regression_known_error():
    """
    Test regression metrics using predictions with a known error.
    """
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])

    metrics = evaluate_regression(y_true, y_pred)

    # Absolute errors are 1, 0 and 1, giving an MAE of 2/3
    assert np.isclose(metrics["MAE"], 2 / 3)

    # Squared errors are 1, 0 and 1, giving RMSE = sqrt(2/3)
    assert np.isclose(metrics["RMSE"], np.sqrt(2 / 3))

    # These predictions give an R-squared value of zero
    assert np.isclose(metrics["R2"], 0.0)
