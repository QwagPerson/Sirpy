from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sirpy.models.abstractModel import AbstractModel
from sirpy.utils.lambdaUtils import null_lambda, add_functions, difference_functions


class AbstractTrainer(ABC):
    """ A base class for all the trainers. It contains the basic methods that every trainer should have.
    A trainer receives a model and trains it. It is used to calculate the parameters of the model.

    Parameters
    ----------
    model : AbstractModel
        The model to be trained.
    args : Any
        Any additional arguments used here to ease inheritance.
    kwargs : Any
        Any additional keyword arguments used here to ease inheritance.

    Attributes
    ----------
    results : Any
        The results of the training process. It can be anything depending on the
        trainer.
    lambda_dict : dict
        A dictionary of the lambda composed functions of the transitions used to calculate
        the gradients of every state.
    metrics : dict
        A dictionary of the metrics used to evaluate the results of the training process.
    """
    def __init__(self, model: AbstractModel, *args: Any, **kwargs: Any) -> None:
        self.model = model
        self.results = None
        self.lambda_dict = None
        self.metrics = {}
        self.compute_lamda_dict()

    @abstractmethod
    def train(self, **kwargs: Any) -> None:
        """ A method that trains the model.
         It should be implemented in the child class."""
        ...

    @abstractmethod
    def solve_ode_system(self) -> Any:
        """ A method that solves the ode system described by the model.
         It should be implemented in the child class."""
        ...

    @abstractmethod
    def calculate_curves(self) -> np.ndarray:
        """ A method that calculates the curves of the model.
        every curve represent the value of a state in time calculated by the model."""
        ...

    @abstractmethod
    def calculate_test_curves(self, **kwargs) -> np.ndarray:
        """ A method that calculates the curves of the model.
        every curve represent the value of a state in time calculated by the model."""
        ...

    def compute_lamda_dict(self) -> None:
        """Goes through the transitions and creates a dictionary of the lambda composed functions of the transitions.
        The dictionary is saved in the lambda_dict attribute and it used to calculated the gradients of every state.
        It has the following structure:
        { state_name: gradient formula - the result of all the lambda functions related to that state composed together}
        """
        self.lambda_dict = {i: null_lambda for i in self.model.states.keys()}
        for i, transition in enumerate(self.model.transitions):
            self.lambda_dict[transition.left] = difference_functions(
                self.lambda_dict[transition.left],
                transition.fun
            )
            # if transition is symmetrical add right else pass
            if transition.symmetrical:
                self.lambda_dict[transition.right] = add_functions(
                    self.lambda_dict[transition.right],
                    transition.fun
                )

    def plot_curves(self) -> plt.Axes:
        """Plots the curves of the model. One per every state.
        Returns
        -------
        plt.Axes
            The axes of the generated plot.
        """
        return pd.DataFrame(self.calculate_curves(), columns=list(self.model.states.keys())).plot(title=self.model.name)

    def plot_train_curves(self) -> None:
        """Make plots of the train curves and the calculated curves. Used to compare the results of the training process.
        """
        # amount of states
        n_states = len(self.model.states.keys())
        n_rows = n_states // 2 + 1
        n_cols = 2

        # make a dataframe from the curves
        curves = pd.DataFrame(self.calculate_curves(), columns=list(self.model.states.keys()))

        # make a dataframe from the train data
        train_data = pd.DataFrame(self.model.train_data, columns=list(self.model.states.keys()))

        # make subplots
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 7))

        # for every ax in axes make a plot of the ith state from curves and from train_data
        for i, ax, col_name in zip(range(axes.size), axes.flatten(), curves.columns):
            curves.iloc[:, i].plot(ax=ax, label="calculated")
            train_data.iloc[:, i].plot(ax=ax, label="Train data")
            ax.set_title(f"Train comparison of State N°{i} - {col_name}")
            ax.legend()

        # make invisible every ax not used
        for i in range(n_states, axes.size):
            axes.flatten()[i].set_visible(False)

    # Make a plot test curves just like the method above
    def plot_test_curves(self) -> None:
        # amount of states
        n_states = len(self.model.states.keys())
        n_rows = n_states // 2 + 1
        n_cols = 2

        # make a dataframe from the curves
        curves = pd.DataFrame(self.calculate_test_curves(), columns=list(self.model.states.keys()))

        # make a dataframe from the train data
        test_data = pd.DataFrame(self.model.test_data, columns=list(self.model.states.keys()))

        # make subplots
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, n_rows * 7))

        # for every ax in axes make a plot of the ith state from curves and from train_data
        for i, ax, col_name in zip(range(axes.size), axes.flatten(), curves.columns):
            curves.iloc[:, i].plot(ax=ax, label="calculated")
            test_data.iloc[:, i].plot(ax=ax, label="Test data")
            ax.set_title(f"Test comparison of State N° State {i} - {col_name}")
            ax.legend()

        # make invisible every ax not used
        for i in range(n_states, axes.size):
            axes.flatten()[i].set_visible(False)

    def get_metrics_report(self):
        """Returns a report of the metrics used to evaluate the results of the training process.
        Returns
        -------
        str
            The report of the metrics.
        """
        report = f"Metric report: \n"
        for metric_name, metric in self.metrics.items():
            report += f"\t{metric_name}: {metric(self.model.test_data, self.calculate_test_curves())}. \n"
        return report
