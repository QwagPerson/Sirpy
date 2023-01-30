# TODO: ADD THIS INTO FORMAT BY CREATING A METRIC OBJECT.
# FOR NOW ITS JUST A BUNCH OF FUNCTIONS

import numpy as np


def MAE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error. The smaller that this is the better.
    Parameters
    ----------
    y_test : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    Returns
    -------
    float
        The mean absolute error.
    """
    return np.mean(np.abs(y_test - y_pred))


def MSE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error. The smaller that this is the better.
    Parameters
    ----------
    y_test : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    Returns
    -------
    float
        The mean squared error.
    Notes
    -----
    Is more sensible to outliers than the MAE.
    """
    return np.mean(np.square(y_test - y_pred))


def RMSE(y_test: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root mean squared error. The smaller that this is the better.
    Parameters
    ----------
    y_test : np.ndarray
        The true values.
    y_pred : np.ndarray
        The predicted values.
    Returns
    -------
    float
        The root mean squared error.
    """
    return np.sqrt(np.mean(np.square(y_test - y_pred)))
