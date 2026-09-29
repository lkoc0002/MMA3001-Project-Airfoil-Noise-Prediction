"""Functions for loading and preparing the Airfoil Self-Noise dataset."""

import pandas as pd
from sklearn.model_selection import train_test_split


COLUMN_NAMES = [
    "frequency",
    "angle_of_attack",
    "chord_length",
    "free_stream_velocity",
    "displacement_thickness",
    "sound_pressure_level",
]


def load_airfoil_data(file_path):
    """
    Load the Airfoil Self-Noise dataset.

    Parameters
    ----------
    file_path : str
        Path or URL to the Airfoil Self-Noise data file.

    Returns
    -------
    data : pandas.DataFrame
        Airfoil dataset with descriptive column names.
    """
    data = pd.read_csv(
        file_path,
        sep="\t",
        names=COLUMN_NAMES
    )

    return data


def prepare_model_data(data, test_size=0.20, random_state=42):
    """
    Prepare training and test data for regression modelling.

    Parameters
    ----------
    data : pandas.DataFrame
        Airfoil Self-Noise dataset.
    test_size : float, optional
        Fraction of observations reserved for testing. The default is 0.20.
    random_state : int, optional
        Random seed used to produce a reproducible train-test split.
        The default is 42.

    Returns
    -------
    X_train : pandas.DataFrame
        Training input variables.
    X_test : pandas.DataFrame
        Testing input variables.
    y_train : pandas.Series
        Training sound pressure levels.
    y_test : pandas.Series
        Testing sound pressure levels.
    """
    # Separate the five engineering inputs from the prediction target
    X = data[
        [
            "frequency",
            "angle_of_attack",
            "chord_length",
            "free_stream_velocity",
            "displacement_thickness",
        ]
    ]

    y = data["sound_pressure_level"]

    # Use a fixed random state so model comparisons use the same observations
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )

    return X_train, X_test, y_train, y_test
