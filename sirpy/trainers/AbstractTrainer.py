from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from matplotlib import pyplot as plt

from abstractModel import AbstractModel


class AbstractTrainer(ABC):
    def __init__(self, model: AbstractModel, *args: Any, **kwargs: Any) -> None:
        self.model = model
        self.results = None
        pass

    @abstractmethod
    def train(self, **kwargs: Any):
        ...

    @abstractmethod
    def solve_ode_system(self):
        ...

    @abstractmethod
    def calculate_curves(self):
        ...

    def plot_curves(self):
        return pd.DataFrame(self.calculate_curves(), columns=list(self.model.states.keys())).plot()

    # make plots of the train curves and the calculated curves
    def plot_train_curves(self):
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
            train_data.iloc[:, i].plot(ax=ax, label="train_data")
            ax.set_title(f"State {i} - {col_name}")
            ax.legend()
