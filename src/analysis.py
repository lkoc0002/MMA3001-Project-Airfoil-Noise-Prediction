"""Functions for engineering sensitivity analysis of regression models."""

import numpy as np
import pandas as pd


def sensitivity_prediction(model, data, variable, points=100):
    """
    Predict sound pressure level while varying one engineering input.

    Parameters
    ----------
    model : regression model
        Fitted regression model used to predict sound pressure level.
    data : pandas.DataFrame
        Input dataset containing the engineering variables.
    variable : str
        Name of the input variable to vary.
    points : int, optional
        Number of values evaluated across the observed variable range.
        The default is 100.

    Returns
    -------
    variable_values : numpy.ndarray
        Values of the selected engineering variable used for prediction.
    predictions : numpy.ndarray
        Predicted sound pressure levels corresponding to each variable value.
    """
    # Use median operating conditions as the reference point
    reference = data.median()

    # Vary only the selected input within its observed dataset range
    variable_values = np.linspace(
        data[variable].min(),
        data[variable].max(),
        points
    )

    # Create repeated reference conditions for the sensitivity analysis
    sensitivity_data = pd.DataFrame(
        [reference.values] * points,
        columns=data.columns
    )

    # Replace the selected input with values spanning its observed range
    sensitivity_data[variable] = variable_values

    predictions = model.predict(sensitivity_data)

    return variable_values, predictions
